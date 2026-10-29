"""PS 02 Early-Warning System - local Streamlit app.

Run:  streamlit run app.py     (or: make app)
Works fully offline from the committed dataset in outputs/ and data/.
Optional: set FIRECRAWL_API_KEY to enable the 'Check for newer filings' button (read-only; never changes scores).
"""
import json, os
from pathlib import Path
import numpy as np, pandas as pd, streamlit as st

ROOT = Path(__file__).parent
O, R = ROOT / "outputs", ROOT / "data" / "reference"
st.set_page_config(page_title="PS02 Early-Warning System", layout="wide")

@st.cache_data
def load():
    need = ["quarterly_scores.csv.gz", "watchlist_latest.csv", "india_eval.json", "polish_metrics.json"]
    miss = [n for n in need if not (O / n).exists()]
    if miss:
        return None
    d = {"q": pd.read_csv(O / "quarterly_scores.csv.gz", parse_dates=["quarter", "fy_end", "available_from"]),
         "w": pd.read_csv(O / "watchlist_latest.csv"),
         "ev": json.load(open(O / "india_eval.json")), "pm": json.load(open(O / "polish_metrics.json")),
         "cases": pd.read_csv(R / "cases.csv"), "peers": pd.read_csv(R / "peers.csv"), "soft": pd.read_csv(R / "soft_signals.csv"),
         "cr": pd.read_csv(O / "india_case_results.csv"), "meta": pd.read_csv(O / "company_meta.csv"),
         "gaps": pd.read_csv(O / "data_gaps_cases.csv"), "ec": pd.read_csv(O / "india_expected_cost_by_quarter.csv"),
         "pcost": pd.read_csv(O / "polish_holdout_cost_table.csv"), "hist": pd.read_csv(O / "watchlist_history_top40.csv", parse_dates=["quarter"])}
    for f in ["shap_values.csv.gz", "provenance_financials.csv.gz"]:
        d[f] = O / f
    return d

D = load()
if D is None:
    st.error("Outputs not found. Run `python run_all.py` (or `make all`) first, then reload.")
    st.stop()
q, W, EV, PM = D["q"], D["w"], D["ev"], D["pm"]
names = D["meta"].set_index("slug").name.to_dict()

st.title("PS 02 - Risk & Fraud Early-Warning System")
st.caption(f"Monitors {EV['watchlist']['n_monitored']} listed Indian non-financial borrowers (frozen universe, as of 2026-10-29); flags a hard-capped top-k = min(40, 5%) each quarter. "
           "Universe = real NSE-listed companies used as a proxy loan book (not simulated). Annual statements mapped to quarters "
           f"with a {EV['avail_lag_days']}-day public-availability lag.")

tabs = st.tabs(["Watch-list", "Company drill-down", "Metrics", "Indian default cases", "Policy & costs", "Provenance & data gaps", "Refresh (optional)"])

# ---------------- Watch-list ----------------
with tabs[0]:
    qs = sorted(q.quarter.unique())
    sel_q = st.select_slider("Quarter", options=qs, value=qs[-1], format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
    d = q[(q.quarter == sel_q) & q.eligible].sort_values("score_full", ascending=False).reset_index(drop=True)
    d["rank"] = d.index + 1; d["company"] = d.slug.map(names)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Borrowers monitored", len(d)); c2.metric("Alert cutoff (score of k-th name)", f"{d.cut_score_full.iloc[0]:.2f}" if len(d) and pd.notna(d.cut_score_full.iloc[0]) else "n/e (pre-coverage)")
    c3.metric("Alerts (cap = min(40, 5%))", f"{int(d.flag_score_full.sum())} / {int(d.alert_cap.iloc[0]) if len(d) else 0}"); c4.metric("Review capacity", 40)
    show_def = st.checkbox("Include the historical defaulter cases in the ranking (backtest view)", value=True)
    if not show_def: d = d[d.group != "defaulter"]
    d["status"] = np.where(d.group == "defaulter", "KNOWN LATER DEFAULTER", "")
    st.dataframe(d.head(40)[["rank", "company", "slug", "group", "status", "fy_label", "score_full", "pd_br0.025", "Z2", "F_score", "M_score", "ret_12m", "months_since_fy_end", "driver_1", "driver_2", "driver_3"]]
                 .rename(columns={"pd_br0.025": "PD (2.5% base-rate assumption)", "fy_label": "latest FY used", "ret_12m": "12m price return"}),
                 width="stretch", height=600, hide_index=True)
    st.download_button("Download this quarter's ranking (CSV)", d.to_csv(index=False), f"watchlist_{pd.Timestamp(sel_q).date()}.csv")

# ---------------- Company drill-down ----------------
with tabs[1]:
    opts = sorted(q.slug.unique(), key=lambda s: (s not in set(D["cases"].slug), s))
    slug = st.selectbox("Company", opts, format_func=lambda s: f"{s} - {names.get(s, '')}")
    g = q[q.slug == slug].sort_values("quarter")
    case = D["cases"][D["cases"].slug == slug]
    if len(case):
        c = case.iloc[0]; st.error(f"Default event {c.event_date}: {c.event_type}  - source: {c.source_url}")
    if g.empty:
        st.info("No scorable quarters.")
    else:
        last = g.iloc[-1]
        a, b, c_, e = st.columns(4)
        a.metric("Latest composite score", f"{last.score_full:.2f}", help="log-odds: model + forensic/market overlay")
        b.metric("Rank (of monitored)", "-" if pd.isna(last.rank_score_full) else int(last.rank_score_full))
        c_.metric("Altman Z''", f"{last.Z2:.2f}" if pd.notna(last.Z2) else "n/a"); e.metric("Piotroski F / Beneish M", f"{last.F_score if pd.notna(last.F_score) else 'n/a'} / {last.M_score:.2f}" if pd.notna(last.M_score) else f"{last.F_score} / n/a")
        ch = g.set_index("quarter")[["score_full", "cut_score_full", "score_model"]].rename(columns={"score_full": "composite score", "cut_score_full": "5% alert cutoff", "score_model": "model-only"})
        st.line_chart(ch)
        qsel = st.selectbox("Explain quarter", list(g.quarter[::-1]), format_func=lambda x: pd.Timestamp(x).strftime("%Y-%m-%d"))
        r = g[g.quarter == qsel].iloc[0]
        st.subheader("Top-3 risk drivers (plain language)")
        for k in ["driver_1", "driver_2", "driver_3"]:
            if isinstance(r[k], str) and r[k]: st.markdown(f"- {r[k]}")
        st.caption(f"Statements used: FY {r.fy_label} (public from {pd.Timestamp(r.available_from).date()}). Driver log-odds: {r.driver_logodds}. Flagged (within cap): {bool(r.flag_score_full)}")
        sh = pd.read_csv(D["shap_values.csv.gz"], parse_dates=["quarter"])
        s = sh[(sh.slug == slug) & (sh.quarter == qsel)]
        if len(s):
            v = s.filter(like="shap_").T.iloc[:, 0].sort_values()
            v.index = v.index.str.replace("shap_", "")
            st.markdown("**SHAP contributions of the financial-ratio model (log-odds, + = riskier)**")
            st.bar_chart(pd.concat([v.head(5), v.tail(8)]))
        with st.expander("Ratios used this quarter"):
            st.dataframe(r.to_frame("value").astype(str), width="stretch")
        soft = D["soft"][D["soft"].slug == slug]
        if len(soft):
            st.subheader("Soft signals (researched, dated, sourced)"); st.dataframe(soft, width="stretch", hide_index=True)
        with st.expander("Provenance of this company's financial data points"):
            pv = pd.read_csv(D["provenance_financials.csv.gz"])
            st.dataframe(pv[pv.slug == slug], width="stretch", hide_index=True)

# ---------------- Metrics ----------------
with tabs[2]:
    st.subheader("Distress model - UCI Polish bankruptcy data (horizon/time-aware split)")
    h = PM["holdout"]
    c = st.columns(5)
    c[0].metric("Holdout PR-AUC (transfer LGBM)", f"{h['pr_auc_transfer_selected']:.3f}")
    c[1].metric("No-skill PR-AUC (= base rate)", f"{h['no_skill_pr_auc']:.3f}")
    c[2].metric("Holdout ROC-AUC", f"{h['roc_auc_transfer_selected']:.3f}")
    c[3].metric("Recall @5% alert (holdout)", f"{h['recall_at_5pct_alert']:.3f}")
    c[4].metric("Altman Z'' PR-AUC", f"{h['pr_auc_altman_z2_standalone']:.3f}")
    st.write(pd.DataFrame({"n": PM["n"], "positives": PM["pos"], "positive rate": PM["pos_rate"]}))
    st.json({"validation PR-AUC": PM["val_pr_auc"], "holdout": h, "threshold": PM["threshold"], "secondary split": PM["secondary_within_5year_split"]}, expanded=False)
    st.subheader("Indian defaulters - recall at a 5% alert rate")
    rec = EV["recall"]
    rows = []
    for v, lab in [("score_model", "Model only"), ("score_model_forensic", "Model + forensic (Z''/F/M)"), ("score_full", "Full composite (+ price, overdue accounts)")] + \
                  ([("score_posthoc_credit", "Post-hoc: + credit rules (in-sample, optimistic)")] if "score_posthoc_credit_recall_4Q_all_cases" in rec else []):
        rows.append({"variant": lab,
                     "recall@4Q (all cases)": f"{rec[f'{v}_recall_4Q_all_cases']['hits']}/{rec[f'{v}_recall_4Q_all_cases']['n']}",
                     "recall@8Q (all cases)": f"{rec[f'{v}_recall_8Q_all_cases']['hits']}/{rec[f'{v}_recall_8Q_all_cases']['n']}",
                     "recall@8Q (scorable)": f"{rec[f'{v}_recall_8Q_scorable']['hits']}/{rec[f'{v}_recall_8Q_scorable']['n']}",
                     "flagged anywhere -8Q..-4Q": f"{rec[f'{v}_flag_in_window_-8Q_to_-4Q']['hits']}/{rec[f'{v}_flag_in_window_-8Q_to_-4Q']['n']}",
                     "peer flag rate": f"{rec[f'{v}_peer_flag_rate_all_quarters']:.1%}", "universe flag rate": f"{rec[f'{v}_universe_flag_rate']:.1%}"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.subheader("Leakage controls"); st.json(EV["leakage_checks"])
    for name, f in [("Sensitivity: 60-day availability lag (LODR results)", "india_eval_lag60.json")]:
        if (O / f).exists():
            s = json.load(open(O / f))["recall"]
            st.caption(f"{name}: full composite recall@4Q {s['score_full_recall_4Q_all_cases']['hits']}/{s['score_full_recall_4Q_all_cases']['n']}, @8Q {s['score_full_recall_8Q_all_cases']['hits']}/{s['score_full_recall_8Q_all_cases']['n']}")

# ---------------- Cases ----------------
with tabs[3]:
    st.subheader("Documented Indian default events (earliest default / D rating found)")
    st.dataframe(D["cases"], width="stretch", hide_index=True,
                 column_config={"source_url": st.column_config.LinkColumn(), "secondary_source": st.column_config.LinkColumn()})
    st.subheader("Backtest result per case")
    st.dataframe(D["cr"], width="stretch", hide_index=True)
    st.subheader("Sector-matched non-defaulting peers"); st.dataframe(D["peers"], width="stretch", hide_index=True)
    st.subheader("Soft signals"); st.dataframe(D["soft"], width="stretch", hide_index=True, column_config={"source_url": st.column_config.LinkColumn()})

# ---------------- Policy ----------------
with tabs[4]:
    st.markdown((ROOT / "reports" / "policy.md").read_text() if (ROOT / "reports" / "policy.md").exists() else "Run the pipeline to generate reports/policy.md")
    st.subheader("Polish holdout - cost of threshold rules (missed default = 20, false alarm = 1)"); st.dataframe(D["pcost"], hide_index=True)
    st.subheader("Model-expected cost on the monitored universe (mean per quarter; PD prior-shifted to an assumed base rate)")
    st.dataframe(D["ec"].groupby(["base_rate", "policy"])[["alerts", "exp_missed_defaults", "exp_false_alarms", "exp_cost"]].mean().round(2).reset_index(), hide_index=True)
    st.subheader("Matched case/peer panel - realised cost"); st.dataframe(pd.DataFrame(EV["matched_panel_cost"]), hide_index=True)

# ---------------- Provenance ----------------
with tabs[5]:
    pv = pd.read_csv(D["provenance_financials.csv.gz"])
    st.write(f"{len(pv):,} financial data points, each with source URL, access time (UTC) and as-of availability date.")
    f = st.text_input("Filter by slug (e.g. JETAIRWAYS)")
    st.dataframe(pv[pv.slug.str.contains(f.upper(), regex=False)] if f else pv.head(2000), width="stretch", hide_index=True)
    st.subheader("Data gaps for the defaulter cases"); st.dataframe(D["gaps"], width="stretch", hide_index=True)
    st.subheader("Company coverage"); st.dataframe(D["meta"], width="stretch", hide_index=True)

# ---------------- Refresh ----------------
with tabs[6]:
    key = os.environ.get("FIRECRAWL_API_KEY")
    st.write("The app runs entirely from the committed dataset. If `FIRECRAWL_API_KEY` is set, you can fetch a company's current public page to see whether "
             "newer annual results exist than those in the dataset. This is read-only: scores change only when you re-run the pipeline.")
    if not key:
        st.info("FIRECRAWL_API_KEY not set - refresh disabled (everything else works).")
    else:
        s2 = st.selectbox("Company to check", sorted(q.slug.unique()), key="rf")
        if st.button("Check for newer filings"):
            import requests, re
            url = D["meta"].set_index("slug").page_url.get(s2)
            try:
                r = requests.post("https://api.firecrawl.dev/v2/scrape", headers={"Authorization": f"Bearer {key}"},
                                  json={"url": url, "formats": ["markdown"]}, timeout=60)
                r.raise_for_status(); md = r.json().get("data", {}).get("markdown", "")
                yrs = sorted(set(re.findall(r"(?:Mar|Jun|Sep|Dec) 20\d\d", md)), key=lambda x: (x[-4:], x[:3]))
                have = q[q.slug == s2].fy_label.max()
                st.write(f"Latest FY in committed dataset: **{have}**; periods seen on live page: {', '.join(yrs[-6:])}")
                (ROOT / "data" / "refresh").mkdir(parents=True, exist_ok=True)
                (ROOT / "data" / "refresh" / f"{s2.replace('&', '_and_')}.md").write_text(md)
            except Exception as ex:  # noqa: BLE001
                st.error(f"Refresh failed: {ex}")
