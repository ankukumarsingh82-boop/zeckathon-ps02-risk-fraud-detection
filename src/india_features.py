"""Turn cached screener snapshots into a firm-year panel of base items, Polish-equivalent ratios,
Altman Z'', Piotroski F, Beneish M, with a provenance record for every base data point.

Availability rule (leakage control): an annual report for a fiscal year ending on date E is treated as
public only from E + 183 days (Companies Act s.96 AGM deadline ~ six months after year end;
SEBI LODR audited results are due within 60 days, so this is conservative).
"""
import glob, json, os, re
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from common import P

AVAIL_LAG_DAYS = 183
MONTHS = {m: i for i, m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], 1)}

def num(x):
    if x is None: return np.nan
    s = str(x).replace(",", "").replace("%", "").strip()
    if s in ("", "-", "--"): return np.nan
    try: return float(s)
    except ValueError: return np.nan

def fy_end(label):
    m = re.match(r"([A-Za-z]{3}) (\d{4})", label)
    if not m: return None
    mo, yr = MONTHS[m.group(1)], int(m.group(2))
    return pd.Timestamp(yr, mo, 1) + pd.offsets.MonthEnd(0)

ROWMAP = {  # (section, row) -> field
    ("profit-loss", "Sales"): "sales", ("profit-loss", "Revenue"): "sales",
    ("profit-loss", "Expenses"): "expenses", ("profit-loss", "Operating Profit"): "op_profit",
    ("profit-loss", "Other Income"): "other_income", ("profit-loss", "Interest"): "interest",
    ("profit-loss", "Depreciation"): "depreciation", ("profit-loss", "Profit before tax"): "pbt",
    ("profit-loss", "Net Profit"): "net_profit",
    ("balance-sheet", "Equity Capital"): "equity_capital", ("balance-sheet", "Reserves"): "reserves",
    ("balance-sheet", "Borrowings"): "borrowings", ("balance-sheet", "Other Liabilities"): "other_liabilities",
    ("balance-sheet", "Total Liabilities"): "total_liabilities_and_equity", ("balance-sheet", "Fixed Assets"): "fixed_assets",
    ("balance-sheet", "CWIP"): "cwip", ("balance-sheet", "Investments"): "investments",
    ("balance-sheet", "Other Assets"): "other_assets", ("balance-sheet", "Total Assets"): "total_assets",
    ("cash-flow", "Cash from Operating Activity"): "cfo",
}
SCHED = {  # (parent, child) -> field
    ("Other Assets", "Inventories"): "inventories", ("Other Assets", "Trade receivables"): "receivables",
    ("Other Assets", "Cash Equivalents"): "cash", ("Other Assets", "Loans n Advances"): "loans_advances",
    ("Other Assets", "Other asset items"): "other_asset_items",
    ("Other Liabilities", "Trade Payables"): "trade_payables",
    ("Borrowings", "Long term Borrowings"): "lt_borrowings", ("Borrowings", "Short term Borrowings"): "st_borrowings",
    ("Borrowings", "Other Borrowings"): "other_borrowings",
    ("Expenses", "Material Cost %"): "material_cost_pct", ("Expenses", "Manufacturing Cost %"): "manufacturing_cost_pct",
    ("Expenses", "Employee Cost %"): "employee_cost_pct", ("Expenses", "Other Cost %"): "other_cost_pct",
}

def load_company(path):
    d = json.load(open(path))
    if "tables" not in d: return None, None, d
    acc = {f["url"]: f["accessed_utc"] for f in d.get("fetches", [])}
    page = d.get("page_url"); acc_page = acc.get(page)
    rows, prov = {}, []
    for (sec, row), field in ROWMAP.items():
        for lab, v in d["tables"].get(sec, {}).get(row, {}).items():
            e = fy_end(lab)
            if e is None: continue
            val = num(v)
            rows.setdefault(lab, {})[field] = val
            prov.append((d["slug"], lab, field, val, f"{page}#{sec}", acc_page))
    cid = d.get("company_id"); cons = "true" if d.get("variant") == "consolidated" else ""
    for (parent, child), field in SCHED.items():
        sch = d.get("schedules", {}).get(parent, {}) or {}
        node = sch.get(child, {}) or {}
        sec = "profit-loss" if parent == "Expenses" else "balance-sheet"
        url = [u for u in acc if "schedules" in u and requests_quote(parent) in u]
        url = url[0] if url else f"https://www.screener.in/api/company/{cid}/schedules/?parent={parent}&section={sec}&consolidated={cons}"
        for lab, v in node.items():
            if fy_end(lab) is None: continue
            val = num(v)
            rows.setdefault(lab, {})[field] = val
            prov.append((d["slug"], lab, field, val, url, acc.get(url)))
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "fy_label"; df = df.reset_index()
    df["fy_end"] = pd.to_datetime(df.fy_label.map(fy_end)) if len(df) else pd.Series(dtype="datetime64[ns]")
    df = df[df.fy_end.notna()]
    if df.empty or "total_assets" not in df: return None, None, d
    df = df.sort_values("fy_end").reset_index(drop=True)
    # drop trailing TTM-style half-year columns (e.g. 'Sep 2021' after 'Mar 2021') that are not full years
    if len(df) >= 2:
        gaps = df.fy_end.diff().dt.days
        keep = ~((gaps < 300) & (df.index == len(df) - 1))
        df = df[keep]
    df["slug"] = d["slug"]; df["name"] = d.get("name"); df["variant"] = d.get("variant")
    df["sector"] = " > ".join(d.get("sector_links", [])[:3]); df["financial_format"] = d.get("financial_format")
    df["available_from"] = df.fy_end + pd.Timedelta(days=AVAIL_LAG_DAYS)
    df["period_months"] = (df.fy_end.dt.year * 12 + df.fy_end.dt.month).diff()
    pv = pd.DataFrame(prov, columns=["slug", "fy_label", "field", "value", "source_url", "accessed_utc"])
    return df, pv, d

def requests_quote(s): return s.replace(" ", "%20")

def sdiv(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    return r.where(np.isfinite(r) & (b != 0))

def add_ratios(df):
    g = df.copy()
    for c in ["inventories","receivables","cash","loans_advances","other_asset_items","trade_payables","lt_borrowings","st_borrowings","other_borrowings","cfo","interest","depreciation","cwip","investments","material_cost_pct","manufacturing_cost_pct","employee_cost_pct","other_cost_pct","op_profit","other_assets","other_liabilities","fixed_assets","net_profit","pbt","sales","reserves","equity_capital","total_assets","borrowings"]:
        if c not in g: g[c] = np.nan
    # Schedule items absent while the 'Other Assets' schedule exists mean a zero balance (e.g. no inventory at an
    # IT-services firm); interest absent/zero means no finance cost. Leaving these NaN would mimic the Polish data's
    # "missing" pattern (rare and distress-associated) and wrongly push debt-free firms up the ranking.
    has_sched = g[["receivables", "cash", "loans_advances", "other_asset_items"]].notna().any(axis=1)
    for c in ["inventories", "receivables", "cash", "loans_advances", "other_asset_items"]:
        g[c] = g[c].where(g[c].notna() | ~has_sched, 0.0)
    g["st_borrowings"] = g.st_borrowings.where(g.st_borrowings.notna() | g.borrowings.isna() | (g.borrowings != 0), 0.0)
    g["lt_borrowings"] = g.lt_borrowings.where(g.lt_borrowings.notna() | g.borrowings.isna() | (g.borrowings != 0), 0.0)
    g["interest"] = g.interest.where(g.interest.notna() | g.pbt.isna(), 0.0)
    g["equity"] = g.equity_capital + g.reserves
    g["TA"] = g.total_assets
    g["TL"] = g.TA - g.equity
    g["CA"] = g.other_assets                       # screener 'Other Assets' = inventories+receivables+cash+loans+other items
    g["STL"] = g.other_liabilities + g.st_borrowings.fillna(0)
    g["LTL"] = g.TL - g.STL
    g["WC"] = g.CA - g.STL
    g["EBIT"] = g.pbt + g.interest.fillna(0)
    g["opp"] = g.op_profit - g.depreciation.fillna(0)  # 'profit on operating activities' (Polish definition)
    s = g.sales
    g["sales_prev"] = g.sales.shift(1)
    R = {}
    R["Attr1"] = sdiv(g.net_profit, g.TA); R["Attr2"] = sdiv(g.TL, g.TA); R["Attr3"] = sdiv(g.WC, g.TA)
    R["Attr4"] = sdiv(g.CA, g.STL); R["Attr6"] = sdiv(g.reserves, g.TA); R["Attr7"] = sdiv(g.EBIT, g.TA)
    R["Attr8"] = sdiv(g.equity, g.TL); R["Attr9"] = sdiv(s, g.TA); R["Attr10"] = sdiv(g.equity, g.TA)
    R["Attr13"] = sdiv(g.pbt + g.depreciation, s); R["Attr16"] = sdiv(g.pbt + g.depreciation, g.TL)
    R["Attr18"] = sdiv(g.pbt, g.TA); R["Attr21"] = sdiv(s, g.sales_prev); R["Attr22"] = sdiv(g.opp, g.TA)
    R["Attr23"] = sdiv(g.net_profit, s); R["Attr26"] = sdiv(g.net_profit + g.depreciation, g.TL)
    R["Attr27"] = sdiv(g.opp, g.interest)
    # zero finance cost: interest cover is effectively unbounded (clipped later to the Polish-train 99th pct) if
    # operating profit is positive, and 0 if not
    R["Attr27"] = R["Attr27"].where(~((g.interest == 0) & g.opp.notna()), np.where(g.opp > 0, 1e6, 0.0)); R["Attr30"] = sdiv(g.TL - g.cash, s)
    R["Attr38"] = sdiv(g.equity + g.LTL, g.TA); R["Attr40"] = sdiv(g.CA - g.inventories - g.receivables, g.STL)
    R["Attr44"] = sdiv(g.receivables * 365, s); R["Attr46"] = sdiv(g.CA - g.inventories, g.STL)
    R["Attr50"] = sdiv(g.CA, g.TL); R["Attr51"] = sdiv(g.STL, g.TA); R["Attr59"] = sdiv(g.LTL, g.equity)
    R["Attr61"] = sdiv(s, g.receivables); R["Attr62"] = sdiv(g.STL * 365, s); R["Attr63"] = sdiv(s, g.STL)
    R["Attr64"] = sdiv(s, g.fixed_assets)
    # irregular (non-12-month) periods make flow ratios incomparable -> growth set missing
    R["Attr21"] = R["Attr21"].where(g.period_months.between(11, 13))
    # Interest cover with zero interest = no debt service burden: leave NaN, model treats as missing
    for k, v in R.items(): g[k] = v
    # ---------- Altman Z'' (Altman 2005 emerging-market / non-manufacturer form, book equity) ----------
    g["Z2"] = 6.56 * sdiv(g.WC, g.TA) + 3.26 * sdiv(g.reserves, g.TA) + 6.72 * sdiv(g.EBIT, g.TA) + 1.05 * sdiv(g.equity, g.TL)
    g["Z2_zone"] = np.select([g.Z2 < 1.1, g.Z2 < 2.6], ["distress", "grey"], "safe"); g.loc[g.Z2.isna(), "Z2_zone"] = "n/a"
    # ---------- Piotroski F (9 binary signals; needs prior year) ----------
    pr = g.shift(1)
    roa = sdiv(g.net_profit, pr.TA); roa_p = sdiv(pr.net_profit, pr.shift(1).TA) if len(g) > 2 else roa * np.nan
    lev = sdiv(g.lt_borrowings, g.TA); lev_p = sdiv(pr.lt_borrowings, pr.TA)
    cr = sdiv(g.CA, g.STL); cr_p = sdiv(pr.CA, pr.STL)
    cogs_pct = (g.material_cost_pct.fillna(0) + g.manufacturing_cost_pct.fillna(0)).where(g.material_cost_pct.notna() | g.manufacturing_cost_pct.notna()) if "material_cost_pct" in g else pd.Series(np.nan, index=g.index)
    g["gross_margin"] = 1 - cogs_pct / 100
    gm_p = g.gross_margin.shift(1)
    at = sdiv(g.sales, pr.TA); at_p = sdiv(pr.sales, pr.shift(1).TA)
    F = pd.DataFrame({
        "F1_roa_pos": roa > 0, "F2_cfo_pos": g.cfo > 0, "F3_droa_pos": roa > roa_p, "F4_cfo_gt_np": g.cfo > g.net_profit,
        "F5_lev_down": lev <= lev_p, "F6_cr_up": cr > cr_p, "F7_no_dilution": g.equity_capital <= pr.equity_capital,
        "F8_gm_up": g.gross_margin > gm_p, "F9_at_up": at > at_p})
    avail = pd.DataFrame({"F1_roa_pos": roa.notna(), "F2_cfo_pos": g.cfo.notna(), "F3_droa_pos": roa.notna() & roa_p.notna(),
        "F4_cfo_gt_np": g.cfo.notna(), "F5_lev_down": lev.notna() & lev_p.notna(), "F6_cr_up": cr.notna() & cr_p.notna(),
        "F7_no_dilution": pr.equity_capital.notna(), "F8_gm_up": g.gross_margin.notna() & gm_p.notna(), "F9_at_up": at.notna() & at_p.notna()})
    for c in F: g[c] = F[c].where(avail[c]).astype(float)
    g["F_n_avail"] = avail.sum(axis=1)
    # scale to 9 when 1-2 components are missing; require >= 7 components
    g["F_score"] = (F.where(avail).sum(axis=1) * 9 / g.F_n_avail.replace(0, np.nan)).where(g.F_n_avail >= 7).round()
    # ---------- Beneish M (8-variable, 1999) ----------
    sga_pct = (g.employee_cost_pct.fillna(0) + g.other_cost_pct.fillna(0)).where(g.employee_cost_pct.notna()) if "employee_cost_pct" in g else pd.Series(np.nan, index=g.index)
    DSRI = sdiv(sdiv(g.receivables, g.sales), sdiv(pr.receivables, pr.sales))
    GMI = sdiv(gm_p, g.gross_margin)
    AQI = sdiv(1 - sdiv(g.CA + g.fixed_assets, g.TA), 1 - sdiv(pr.CA + pr.fixed_assets, pr.TA))
    SGI = sdiv(g.sales, pr.sales)
    DEPI = sdiv(sdiv(pr.depreciation, pr.depreciation + pr.fixed_assets), sdiv(g.depreciation, g.depreciation + g.fixed_assets))
    SGAI = sdiv(sga_pct, sga_pct.shift(1))
    LVGI = sdiv(sdiv(g.STL + g.lt_borrowings.fillna(0), g.TA), sdiv(pr.STL + pr.lt_borrowings.fillna(0), pr.TA))
    TATA = sdiv(g.net_profit - g.cfo, g.TA)
    comp = pd.DataFrame(dict(DSRI=DSRI, GMI=GMI, AQI=AQI, SGI=SGI, DEPI=DEPI, SGAI=SGAI, LVGI=LVGI, TATA=TATA))
    comp = comp.where(g.period_months.between(11, 13))
    core_ok = comp[["SGI", "TATA"]].notna().all(axis=1) & (comp.notna().sum(axis=1) >= 5)
    g["M_n_imputed"] = comp.isna().sum(axis=1)
    neutral = dict(DSRI=1, GMI=1, AQI=1, SGI=1, DEPI=1, SGAI=1, LVGI=1, TATA=0)
    cf = comp.fillna(neutral)
    cf = cf.clip(lower=-5, upper=10)   # damp extreme index values from tiny denominators
    M = (-4.84 + 0.920 * cf.DSRI + 0.528 * cf.GMI + 0.404 * cf.AQI + 0.892 * cf.SGI + 0.115 * cf.DEPI
         - 0.172 * cf.SGAI + 4.679 * cf.TATA - 0.327 * cf.LVGI)
    g["M_score"] = M.where(core_ok)
    for c in comp: g["M_" + c] = comp[c]
    return g

def build():
    frames, provs, meta = [], [], []
    for p in sorted(glob.glob(P("data", "raw", "screener", "*.json"))):
        df, pv, d = load_company(p)
        meta.append({"slug": d["slug"], "name": d.get("name"), "variant": d.get("variant"), "page_url": d.get("page_url"),
                     "financial_format": d.get("financial_format"), "n_years": 0 if df is None else len(df), "error": d.get("error"),
                     "skip_reason": None if df is not None and not df.empty else ("financial-format statements" if d.get("financial_format") else "no parsable annual statements on source page"),
                     "sector": " > ".join(d.get("sector_links", [])[:3])})
        if df is None or df.empty: continue
        frames.append(add_ratios(df)); provs.append(pv)
    panel = pd.concat(frames, ignore_index=True)
    prov = pd.concat(provs, ignore_index=True)
    prov["fy_end"] = prov.fy_label.map(fy_end)
    prov["as_of_available_date"] = (prov.fy_end + pd.Timedelta(days=AVAIL_LAG_DAYS)).dt.date
    prov["source"] = "screener.in (compiled from company annual reports / exchange filings)"
    return panel, prov, pd.DataFrame(meta)

if __name__ == "__main__":
    panel, prov, meta = build()
    panel.to_parquet(P("data", "processed", "india_firm_years.parquet")) if False else panel.to_csv(P("data", "processed", "india_firm_years.csv"), index=False)
    meta.to_csv(P("data", "processed", "india_company_meta.csv"), index=False)
    print(panel.shape, prov.shape, meta.financial_format.sum())
