"""Quarterly early-warning scoring of the Indian panel + universe, SHAP drivers, evaluation, cost and watch-list.

Leakage controls enforced here:
  * a fiscal year's statements are used only from fy_end + AVAIL_LAG_DAYS (default 183) onward;
  * case quarters on/after the event date are never scored;
  * the transfer model is trained on Polish data only; nothing is fitted on Indian outcomes;
    the only Indian-data statistic is the per-quarter 95th percentile alert cutoff (outcome-free).
"""
import glob, json, os, pickle, warnings
import numpy as np, pandas as pd, shap
warnings.filterwarnings("ignore")
from common import P, FEATS, TRANSFER_FEATURES, COST_FN, COST_FP, ALERT_RATE, Prep  # noqa: F401
from india_features import build

LAG = int(os.environ.get("AVAIL_LAG_DAYS", 183))
TAG = os.environ.get("RUN_TAG", "")              # sensitivity runs write *_<tag> files
QUARTERS = pd.date_range("2013-03-31", "2026-09-30", freq="QE")
OVERLAY = {  # pre-specified prior overlay, in log-odds points (documented in report; not fitted)
    "piotroski_F<=2": 0.5, "piotroski_F=3": 0.25, "beneish_M>-1.78": 0.3, "altman_Z''<1.1": 0.3,
    "price_12m<-50%": 0.7, "price_12m<-30%": 0.35, "accounts_overdue_18m": 0.5}
BASE_RATES = [0.01, 0.025, 0.05]  # assumed Indian annual default base rates (assumption, not sourced)
REVIEW_CAPACITY = 40
FORENSIC = ["piotroski_F<=2", "piotroski_F=3", "beneish_M>-1.78", "altman_Z''<1.1"]
MARKET = ["price_12m<-50%", "price_12m<-30%", "accounts_overdue_18m"]

def cp_ci(k, n, a=0.05):
    """Clopper-Pearson exact 95% interval for a binomial proportion."""
    from scipy.stats import beta
    if n == 0: return [None, None]
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1)); hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]

def load_prices():
    out = {}
    for p in glob.glob(P("data", "raw", "screener", "*.json")):
        d = json.load(open(p))
        pr = d.get("price") or []
        if pr:
            s = pd.Series([float(v) for _, v in pr], index=pd.to_datetime([t for t, _ in pr])).sort_index()
            out[d["slug"]] = s[~s.index.duplicated()]
    return out

def ret12(s, q):
    if s is None: return np.nan
    a = s[:q]; b = s[:q - pd.Timedelta(days=365)]
    if a.empty or b.empty or (q - a.index[-1]).days > 60 or (q - pd.Timedelta(days=365) - b.index[-1]).days > 60: return np.nan
    return a.iloc[-1] / b.iloc[-1] - 1 if b.iloc[-1] > 0 else np.nan

def quarterly_frame(panel, prices, cases):
    panel = panel.copy(); panel["available_from"] = panel.fy_end + pd.Timedelta(days=LAG)
    ev = dict(zip(cases.slug, cases.event_date))
    rows = []
    for slug, g in panel.groupby("slug"):
        g = g.sort_values("fy_end")
        for q in QUARTERS:
            if slug in ev and q >= ev[slug]: break
            av = g[g.available_from <= q]
            if av.empty: continue
            r = av.iloc[-1]
            rows.append({"slug": slug, "quarter": q, "fy_label": r.fy_label, "fy_end": r.fy_end, "available_from": r.available_from,
                         "row_id": r.name, "months_since_fy_end": (q - r.fy_end).days / 30.44,
                         "ret_12m": ret12(prices.get(slug), q)})
    qf = pd.DataFrame(rows)
    return qf.join(panel.drop(columns=["slug", "fy_label", "fy_end", "available_from"]), on="row_id")

def overlay_terms(df):
    t = pd.DataFrame(index=df.index)
    t["piotroski_F<=2"] = np.where(df.F_score <= 2, OVERLAY["piotroski_F<=2"], 0.0)
    t["piotroski_F=3"] = np.where(df.F_score == 3, OVERLAY["piotroski_F=3"], 0.0)
    t["beneish_M>-1.78"] = np.where(df.M_score > -1.78, OVERLAY["beneish_M>-1.78"], 0.0)
    t["altman_Z''<1.1"] = np.where(df.Z2 < 1.1, OVERLAY["altman_Z''<1.1"], 0.0)
    t["price_12m<-50%"] = np.where(df.ret_12m < -0.5, OVERLAY["price_12m<-50%"], 0.0)
    t["price_12m<-30%"] = np.where((df.ret_12m < -0.3) & ~(df.ret_12m < -0.5), OVERLAY["price_12m<-30%"], 0.0)
    t["accounts_overdue_18m"] = np.where(df.months_since_fy_end > 18, OVERLAY["accounts_overdue_18m"], 0.0)
    return t

def explain_overlay(k, r):
    return {
        "piotroski_F<=2": f"Piotroski F-score only {r.F_score:.0f}/9: profitability, cash flow, leverage and efficiency are mostly deteriorating",
        "piotroski_F=3": f"Piotroski F-score {r.F_score:.0f}/9: more fundamentals worsening than improving",
        "beneish_M>-1.78": f"Beneish M-score {r.M_score:.2f} is above -1.78: accrual/receivable patterns resemble earnings manipulators - check revenue recognition and receivables",
        "altman_Z''<1.1": f"Altman Z'' {r.Z2:.2f} is in the distress zone (<1.1): thin working capital, low retained earnings and/or weak EBIT relative to liabilities",
        "price_12m<-50%": f"Share price fell {-100*r.ret_12m:.0f}% over 12 months: the market is pricing serious stress",
        "price_12m<-30%": f"Share price fell {-100*r.ret_12m:.0f}% over 12 months",
        "accounts_overdue_18m": f"Latest audited accounts are {r.months_since_fy_end:.0f} months old (FY {r.fy_label}): annual results appear delayed - a classic pre-default red flag",
    }[k]

def explain_feature(f, val, pct):
    lab, mono = TRANSFER_FEATURES[f]
    if pd.isna(val): return f"{lab}: not reported in the filings (imputed with the training median) - verify manually"
    if mono == 0:
        return f"{lab} is unusual at {val:.2f} ({pct:.0f}th percentile of monitored borrowers); this raises the distress score"
    worse = pct if mono > 0 else 100 - pct
    side = "high" if mono > 0 else "low"
    return f"{lab} is {side} at {val:.2f} (worse than {worse:.0f}% of monitored borrowers); this raises the distress score"

def main():
    model = pickle.load(open(P("outputs", "transfer_model.pkl"), "rb"))
    prep, gb, iso = model["prep"], model["gb"], model["iso"]
    panel, prov, meta = build()
    cases = pd.read_csv(P("data", "reference", "cases.csv"), parse_dates=["event_date"])
    peers = pd.read_csv(P("data", "reference", "peers.csv"))
    # frozen universe (data/reference/universe_frozen.csv, as of 2026-10-29): later scrapes cannot change the backtest
    frozen = pd.read_csv(P("data", "reference", "universe_frozen.csv"))
    uni = list(frozen[frozen.status == "ok"].slug)
    prices = load_prices()
    keep_slugs = set(uni) | set(cases.slug) | set(peers.slug)
    panel = panel[panel.slug.isin(keep_slugs)]
    qf = quarterly_frame(panel, prices, cases)
    core = set(cases[cases.track == "core"].slug)
    qf["group"] = np.where(qf.slug.isin(core), "defaulter", np.where(qf.slug.isin(peers.slug), "peer",
                  np.where(qf.slug.isin(uni), "universe", "other")))
    qf.loc[qf.slug.isin(cases[cases.track == "financial"].slug), "group"] = "financial_case"
    qf["financial_format"] = qf.financial_format.fillna(False).astype(bool)
    X = prep.tree(qf[FEATS])
    qf["model_logit"] = gb.booster_.predict(X, raw_score=True)
    qf["model_pd_polish"] = iso.predict(gb.predict_proba(X)[:, 1])
    sv = shap.TreeExplainer(gb.booster_).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    terms = overlay_terms(qf)
    qf["overlay_forensic"] = terms[FORENSIC].sum(axis=1); qf["overlay_market"] = terms[MARKET].sum(axis=1)
    qf["score_model"] = qf.model_logit
    qf["score_model_forensic"] = qf.model_logit + qf.overlay_forensic
    qf["score_full"] = qf.score_model_forensic + qf.overlay_market
    # POST-HOC variant (added after seeing headline results; in-sample, reported separately as optimistic):
    # standard bank covenant-style rules - interest cover < 1.5x, debt / EBITDA > 5x (or debt with EBITDA <= 0)
    debt_ebitda = qf.borrowings / qf.op_profit
    qf["rule_icr_lt_1.5"] = (qf.Attr27 < 1.5).astype(float) * 0.5
    qf["rule_debt_ebitda_gt_5"] = (((debt_ebitda > 5) | ((qf.op_profit <= 0) & (qf.borrowings > 0)))).astype(float) * 0.5
    qf["debt_to_ebitda"] = debt_ebitda
    qf["score_posthoc_credit"] = qf.score_full + qf["rule_icr_lt_1.5"] + qf["rule_debt_ebitda_gt_5"]
    elig = (~qf.financial_format) & qf.group.isin(["defaulter", "peer", "universe"])
    if os.environ.get("UNIVERSE") == "totalmarket":   # sensitivity: large/mid-cap book (Nifty Total Market members only)
        tm = set(pd.read_csv(P("data", "reference", "nifty_total_market_list.csv")).iloc[:, 2])
        elig &= qf.slug.isin(tm) | qf.group.isin(["defaulter", "peer"])
    qf["eligible"] = elig
    n_q = qf[elig & qf.group.isin(["universe", "peer"])].groupby("quarter").size()
    full_cov = n_q[n_q >= 300].index.min()
    variants = ["score_model", "score_model_forensic", "score_full", "score_posthoc_credit"]
    # Hard alert cap per quarter: k = min(REVIEW_CAPACITY, floor(5% x eligible N)). Quarters before full universe
    # coverage (fewer than 300 universe/peer names scored) are NOT evaluable: no alerts are raised there.
    n_elig = qf[elig].groupby("quarter").size()
    cap = np.minimum(REVIEW_CAPACITY, np.floor(ALERT_RATE * n_elig)).astype(int)
    cap[cap.index < full_cov] = 0
    qf["n_eligible"] = qf.quarter.map(n_elig); qf["alert_cap"] = qf.quarter.map(cap).fillna(0).astype(int)
    qf["evaluable_quarter"] = qf.quarter >= full_cov
    for v in variants:
        qf["rank_" + v] = qf[elig].groupby("quarter")[v].rank(ascending=False, method="first")
        qf["flag_" + v] = elig & (qf["rank_" + v] <= qf.alert_cap)
        cut = qf[qf["flag_" + v]].groupby("quarter")[v].min()
        qf["cut_" + v] = qf.quarter.map(cut)
    pv = model["val_pos_rate"]; lo = lambda p: np.log(p / (1 - p))
    for br in BASE_RATES:
        z = lo(np.clip(qf.model_pd_polish, 1e-4, 1 - 1e-4)) + lo(br) - lo(pv) + qf.overlay_forensic + qf.overlay_market
        qf[f"pd_br{br}"] = 1 / (1 + np.exp(-z))
    pct = (qf[elig].groupby("quarter")[FEATS].rank(pct=True) * 100).reindex(qf.index)
    drivers = []
    for i in range(len(qf)):
        r = qf.iloc[i]
        contrib = {f: sv[i, j] for j, f in enumerate(FEATS)}
        contrib.update({k: terms.iloc[i][k] for k in OVERLAY})
        top = sorted([(k, v) for k, v in contrib.items() if v > 0], key=lambda kv: -kv[1])[:3]
        txt = [explain_feature(k, r[k], pct.iloc[i][k] if not pd.isna(pct.iloc[i][k]) else 50) if k in FEATS else explain_overlay(k, r) for k, _ in top]
        txt += [""] * (3 - len(txt))
        drivers.append({"driver_1": txt[0], "driver_2": txt[1], "driver_3": txt[2],
                        "driver_keys": ";".join(k for k, _ in top), "driver_logodds": ";".join(f"{v:.2f}" for _, v in top)})
    qf = pd.concat([qf.reset_index(drop=True), pd.DataFrame(drivers)], axis=1)
    shap_df = pd.DataFrame(sv, columns=["shap_" + f for f in FEATS]); shap_df[["slug", "quarter"]] = qf[["slug", "quarter"]]
    shap_df["base_value"] = float(np.ravel(shap.TreeExplainer(gb.booster_).expected_value)[0])
    # ---- evaluation ----
    res = {"avail_lag_days": LAG, "full_coverage_from": str(full_cov.date()), "overlay_logodds": OVERLAY, "alert_rate": ALERT_RATE}
    case_rows = []
    for _, c in cases.iterrows():
        g = qf[qf.slug == c.slug].set_index("quarter")
        row = {"slug": c.slug, "company": c.company, "event_date": c.event_date.date(), "track": c.track, "sector_group": c.sector_group}
        for L, months in [("4Q", 12), ("8Q", 24)]:
            qt = QUARTERS[QUARTERS <= c.event_date - pd.DateOffset(months=months)].max()
            row[f"q_{L}"] = qt.date()
            if qt in g.index:
                rr = g.loc[qt]; row[f"fy_used_{L}"] = rr.fy_label; row[f"available_from_{L}"] = str(pd.Timestamp(rr.available_from).date())
                for v in variants:
                    row[f"{v}_{L}"] = round(float(rr[v]), 3)
                    row[f"flag_{v}_{L}"] = bool(rr["flag_" + v]) if c.track == "core" else None
                    row[f"rank_{v}_{L}"] = None if pd.isna(rr["rank_" + v]) else int(rr["rank_" + v])
                row[f"n_ranked_{L}"] = int(qf[qf.eligible & (qf.quarter == qt)].shape[0])
                row[f"evaluable_{L}"] = bool(rr.evaluable_quarter); row[f"alert_cap_{L}"] = int(rr.alert_cap)
            else:
                row[f"fy_used_{L}"] = None; row[f"evaluable_{L}"] = False
        w = g[(g.index >= c.event_date - pd.DateOffset(months=24)) & (g.index <= c.event_date - pd.DateOffset(months=12))]
        for v in variants:
            row[f"window_flag_{v}"] = bool(w["flag_" + v].any()) if len(w) and c.track == "core" else None
        fl = g[g["flag_score_full"]] if c.track == "core" else g.iloc[0:0]
        row["first_flag_quarter_full"] = fl.index.min().date() if len(fl) else None
        row["first_flag_lead_quarters_full"] = int(round((c.event_date - fl.index.min()).days / 91.3)) if len(fl) else None
        q4 = pd.Timestamp(row["q_4Q"])
        row["drivers_at_4Q"] = " | ".join(str(g.loc[q4][k]) for k in ["driver_1", "driver_2", "driver_3"]) if q4 in g.index else ""
        case_rows.append(row)
    cr = pd.DataFrame(case_rows)
    corec = cr[cr.track == "core"]
    summ = {}
    for v in variants:
        for L in ["4Q", "8Q"]:
            s = corec[f"flag_{v}_{L}"] if f"flag_{v}_{L}" in corec else pd.Series(dtype=object)
            ev_ok = corec.get(f"evaluable_{L}", pd.Series(False, index=corec.index)) == True  # noqa: E712
            scor = int(ev_ok.sum()); hit = int((s == True).sum())  # noqa: E712
            summ[f"{v}_recall_{L}_all_cases"] = {"hits": hit, "n": len(corec), "recall": hit / len(corec), "ci95": cp_ci(hit, len(corec))}
            summ[f"{v}_recall_{L}_scorable"] = {"hits": hit, "n": scor, "recall": (hit / scor) if scor else None, "ci95": cp_ci(hit, scor)}
        s = corec[f"window_flag_{v}"]
        summ[f"{v}_flag_in_window_-8Q_to_-4Q"] = {"hits": int((s == True).sum()), "n": int(s.notna().sum())}  # noqa: E712
        summ[f"{v}_peer_flag_rate_all_quarters"] = float(qf[(qf.group == "peer") & qf.eligible]["flag_" + v].mean())
        summ[f"{v}_universe_flag_rate"] = float(qf[(qf.group == "universe") & qf.eligible]["flag_" + v].mean())
    res["recall"] = summ
    lab = []
    for L, months in [("4Q", 12), ("8Q", 24)]:
        for _, c in cases[cases.track == "core"].iterrows():
            qt = QUARTERS[QUARTERS <= c.event_date - pd.DateOffset(months=months)].max()
            ps = peers[peers.matched_to.str.contains(c.slug, regex=False)].slug
            for s, y in [(c.slug, 1)] + [(p, 0) for p in ps]:
                r = qf[(qf.slug == s) & (qf.quarter == qt)]
                lab.append({"lead": L, "case": c.slug, "slug": s, "y": y, "quarter": qt.date(), "scorable": len(r) > 0,
                            **{f"flag_{v}": (bool(r.iloc[0]["flag_" + v]) if len(r) else False) for v in variants}})
    lab = pd.DataFrame(lab)
    cost_rows = []
    for L in ["4Q", "8Q"]:
        for v in variants:
            d = lab[lab.lead == L]
            tp = int(((d.y == 1) & d["flag_" + v]).sum()); fn = int(((d.y == 1) & ~d["flag_" + v]).sum())
            fp = int(((d.y == 0) & d["flag_" + v]).sum()); tn = int(((d.y == 0) & ~d["flag_" + v]).sum())
            cost_rows.append({"lead": L, "variant": v, "TP": tp, "FN": fn, "FP": fp, "TN": tn,
                              "cost_units": COST_FN * fn + COST_FP * fp, "cost_flag_nobody": COST_FN * (tp + fn),
                              "calc": f"{COST_FN} x {fn} missed + {COST_FP} x {fp} false alarms = {COST_FN*fn + COST_FP*fp}"})
    res["matched_panel_cost"] = cost_rows
    latest = qf[qf.eligible & (qf.group != "defaulter")].quarter.max()
    ec = []
    for br in BASE_RATES:
        col = f"pd_br{br}"
        for qq, d in qf[qf.eligible & qf.quarter.between(full_cov, latest)].groupby("quarter"):
            p = d[col].values; n = len(p); order = np.argsort(-d.score_full.values)
            k = int(min(REVIEW_CAPACITY, np.floor(ALERT_RATE * n)))
            pol = {"flag_nobody": np.zeros(n, bool), "capped_top_k": np.isin(np.arange(n), order[:k]),
                   "bayes_PD>=1/21": p >= COST_FP / (COST_FP + COST_FN)}
            for name, f in pol.items():
                ec.append({"base_rate": br, "quarter": qq.date(), "n": n, "policy": name, "alerts": int(f.sum()),
                           "exp_missed_defaults": float(p[~f].sum()), "exp_false_alarms": float((1 - p[f]).sum()),
                           "exp_cost": float(COST_FN * p[~f].sum() + COST_FP * (1 - p[f]).sum())})
    ec = pd.DataFrame(ec)
    res["expected_cost_latest_quarter"] = ec[ec.quarter == latest.date()].to_dict("records")
    res["expected_cost_mean_over_quarters"] = ec.groupby(["base_rate", "policy"])[["alerts", "exp_missed_defaults", "exp_false_alarms", "exp_cost"]].mean().reset_index().to_dict("records")
    W = qf[qf.eligible & (qf.quarter == latest) & (qf.group != "defaulter")].copy()
    W = W.sort_values("score_full", ascending=False).reset_index(drop=True); W["rank"] = W.index + 1
    W["company"] = W.slug.map(meta.set_index("slug").name.to_dict())
    W["pd_assumed_2.5pct_base"] = W["pd_br0.025"]
    W["action"] = np.select([W.flag_score_full & (W["rank"] <= 10), W.flag_score_full, W["rank"] <= 2 * REVIEW_CAPACITY],
                            ["ESCALATE: senior credit review within 2 weeks", "REVIEW: analyst deep-dive this quarter", "WATCH: monthly price/rating/pledge monitoring"], "Routine")
    res["watchlist"] = {"quarter": str(latest.date()), "n_monitored": len(W), "n_alerts": int(W.flag_score_full.sum()), "alert_cap": int(W.alert_cap.iloc[0]), "review_capacity": REVIEW_CAPACITY}
    res["leakage_checks"] = {
        "feature_rows_available_before_quarter": bool((qf.available_from <= qf.quarter).all()),
        "no_case_quarter_on_or_after_event": bool(all((qf[qf.slug == c.slug].quarter < c.event_date).all() for _, c in cases.iterrows())),
        "model_trained_on_polish_only": True, "preprocessing_fit_on_polish_train_only": True,
        "threshold_uses_no_outcomes": True, "avail_lag_days": LAG,
        "n_quarter_rows": len(qf), "n_companies_scored": int(qf.slug.nunique())}
    gaps = []
    for _, c in cases.iterrows():
        g = panel[panel.slug == c.slug]
        pre = g[g.fy_end + pd.Timedelta(days=LAG) <= c.event_date]
        gaps.append({"slug": c.slug, "n_fy_on_screener": len(g), "first_fy": g.fy_label.iloc[0] if len(g) else None,
                     "n_fy_public_before_event": len(pre), "has_3y_pre_event": len(pre) >= 3,
                     "missing_feature_share_pre_event": round(float(pre[FEATS].isna().mean().mean()), 3) if len(pre) else None,
                     "missing_fields_pre_event": ";".join(sorted(set(f for f in FEATS if len(pre) and pre[f].isna().any()))),
                     "F_score_missing_years": int(pre.F_score.isna().sum()) if len(pre) else None,
                     "M_score_missing_years": int(pre.M_score.isna().sum()) if len(pre) else None})
    res["data_gaps_cases"] = gaps
    soft = pd.read_csv(P("data", "reference", "soft_signals.csv"), parse_dates=["signal_date"])
    sl = []
    for _, c in cases.iterrows():
        ss = soft[soft.slug == c.slug]
        pre = ss[ss.signal_date < c.event_date]; pre4 = ss[ss.signal_date <= c.event_date - pd.DateOffset(months=12)]
        sl.append({"slug": c.slug, "event_date": str(c.event_date.date()), "n_signals": len(ss), "n_before_event": len(pre),
                   "n_at_least_4Q_before": len(pre4), "earliest_signal": str(pre.signal_date.min().date()) if len(pre) else None,
                   "lead_months_earliest": round((c.event_date - pre.signal_date.min()).days / 30.44, 1) if len(pre) else None})
    res["soft_signal_leads"] = sl
    ev_q = qf[qf.eligible & qf.evaluable_quarter].groupby("quarter")
    res["alert_cap_check"] = {"max_alerts_any_quarter": int(ev_q.flag_score_full.sum().max()),
                              "max_alert_rate_any_quarter": float(ev_q.flag_score_full.mean().max()),
                              "n_evaluable_quarters": int(ev_q.ngroups), "cap_rule": f"min({REVIEW_CAPACITY}, floor({ALERT_RATE} x N))"}
    res["universe_size_by_quarter"] = {str(k.date()): int(v) for k, v in n_elig.items()}
    res["case_results"] = json.loads(cr.to_json(orient="records", date_format="iso"))
    suf = f"_{TAG}" if TAG else ""
    json.dump(res, open(P("outputs", f"india_eval{suf}.json"), "w"), indent=2, default=str)
    cr.to_csv(P("outputs", f"india_case_results{suf}.csv"), index=False)
    lab.to_csv(P("outputs", f"india_matched_panel{suf}.csv"), index=False)
    if not TAG:
        keep = ["slug", "quarter", "group", "eligible", "fy_label", "fy_end", "available_from", "months_since_fy_end", "ret_12m", "Z2", "Z2_zone", "F_score", "M_score",
                "model_logit", "model_pd_polish", "overlay_forensic", "overlay_market", "score_model", "score_model_forensic", "score_full",
                "cut_score_full", "flag_score_full", "rank_score_full", "n_eligible", "alert_cap", "evaluable_quarter", "flag_score_model", "flag_score_model_forensic", "score_posthoc_credit", "flag_score_posthoc_credit", "debt_to_ebitda", "pd_br0.01", "pd_br0.025", "pd_br0.05",
                "driver_1", "driver_2", "driver_3", "driver_keys", "driver_logodds"] + FEATS
        qf[keep].to_csv(P("outputs", "quarterly_scores.csv.gz"), index=False, compression="gzip")
        shap_df.to_csv(P("outputs", "shap_values.csv.gz"), index=False, compression="gzip")
        W[["rank", "slug", "company", "sector", "fy_label", "score_full", "score_model", "pd_assumed_2.5pct_base", "Z2", "F_score", "M_score", "ret_12m",
           "months_since_fy_end", "flag_score_full", "action", "driver_1", "driver_2", "driver_3"]].to_csv(P("outputs", "watchlist_latest.csv"), index=False)
        hist = qf[qf.eligible & (qf.rank_score_full <= REVIEW_CAPACITY)][["quarter", "rank_score_full", "slug", "group", "score_full", "driver_1", "driver_2", "driver_3"]]
        hist.sort_values(["quarter", "rank_score_full"]).to_csv(P("outputs", "watchlist_history_top40.csv"), index=False)
        ec.to_csv(P("outputs", "india_expected_cost_by_quarter.csv"), index=False)
        prov.to_csv(P("outputs", "provenance_financials.csv.gz"), index=False, compression="gzip")
        pd.DataFrame(gaps).to_csv(P("outputs", "data_gaps_cases.csv"), index=False)
        meta.to_csv(P("outputs", "company_meta.csv"), index=False)
    print(json.dumps({k: res[k] for k in ["recall", "watchlist", "leakage_checks", "full_coverage_from"]}, indent=1, default=str))
    cols = [c for c in ["slug", "event_date", "q_4Q", "fy_used_4Q", "score_full_4Q", "flag_score_full_4Q", "rank_score_full_4Q", "fy_used_8Q", "flag_score_full_8Q", "window_flag_score_full", "first_flag_lead_quarters_full"] if c in cr]
    print(cr[cols].to_string())

if __name__ == "__main__":
    main()
