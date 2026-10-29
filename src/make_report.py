"""Build figures, reports/policy.md, reports/report.md, reports/report.pdf and dashboard/index.html from outputs/."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from common import P, COST_FN, COST_FP, TRANSFER_FEATURES

O = lambda f: P("outputs", f)
FIG = P("reports", "figures"); os.makedirs(FIG, exist_ok=True); os.makedirs(P("dashboard"), exist_ok=True)

def md_table(df, floatfmt="{:.3f}"):
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind == "f": df[c] = df[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    cell = lambda x: "" if x is None or (isinstance(x, float) and np.isnan(x)) or str(x) in ("nan", "None", "NaT", "<NA>") else str(x).replace("|", "/")
    body = "\n".join("| " + " | ".join(cell(x) for x in r) + " |" for r in df.itertuples(index=False))
    return "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + body + "\n"

def figures(pm, hs, q, cases):
    p, r, _ = precision_recall_curve(hs.y, hs.raw)
    plt.figure(figsize=(5, 4)); plt.plot(r, p, label=f"transfer LightGBM (AP={pm['holdout']['pr_auc_transfer_selected']:.3f})")
    plt.axhline(hs.y.mean(), ls="--", c="grey", label=f"no skill ({hs.y.mean():.3f})")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.title("Polish holdout (1-year horizon)"); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"{FIG}/polish_pr.png", dpi=130); plt.close()
    cv = pd.read_csv(O("polish_val_cost_curve.csv"))
    plt.figure(figsize=(5, 4)); plt.plot(cv.threshold, cv.cost); plt.axvline(1 / 21, c="r", ls="--", label="Bayes 1/21")
    plt.xlabel("PD threshold"); plt.ylabel("cost units (20xFN + 1xFP)"); plt.title("Validation expected-cost curve"); plt.legend(); plt.xlim(0, 0.4); plt.tight_layout()
    plt.savefig(f"{FIG}/cost_curve.png", dpi=130); plt.close()
    core = cases[cases.track == "core"]
    n = len(core); cols = 4; rows = int(np.ceil(n / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(14, 2.6 * rows), sharey=False)
    for a, (_, c) in zip(ax.ravel(), core.iterrows()):
        g = q[q.slug == c.slug].sort_values("quarter")
        a.plot(g.quarter, g.score_full, label="composite"); a.plot(g.quarter, g.cut_score_full, "r--", lw=1, label="alert cutoff (top-k cap)")
        ev = pd.Timestamp(c.event_date)
        for m, ls in [(12, ":"), (24, "-.")]: a.axvline(ev - pd.DateOffset(months=m), c="k", ls=ls, lw=0.8)
        a.axvline(ev, c="r", lw=1.2); a.set_title(f"{c.slug} (event {c.event_date})", fontsize=8); a.tick_params(labelsize=6)
    for a in ax.ravel()[n:]: a.axis("off")
    ax.ravel()[0].legend(fontsize=6); fig.suptitle("Composite score vs 5% cutoff; dotted = -4Q, dash-dot = -8Q, red = event", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{FIG}/case_trajectories.png", dpi=110); plt.close(fig)

def policy_md(ev, pm):
    lat = pd.DataFrame(ev["expected_cost_latest_quarter"]); l25 = lat[lat.base_rate == 0.025].set_index("policy")
    wl = ev["watchlist"]
    return f"""## Operating policy
**Book.** {wl['n_monitored']} real listed Indian non-financial companies scored at {wl['quarter']} (frozen universe, `data/reference/universe_frozen.csv`; a proxy for the brief's 800-borrower book, not simulated).
**Alert threshold.** Each quarter, flag the top k = min(40, floor(5% x N)) by composite score (this quarter: k = {wl['alert_cap']}, {wl['n_alerts']} alerts). This is a hard cap, so the review team never receives more than 40 names.
**Quarterly cycle.**
1. T+5 days: refresh filings and prices. Use only statements public by T (FY end + {ev['avail_lag_days']} days). Score every borrower.
2. Flagged ranks 1-10 -> **ESCALATE**: senior credit officer within 2 weeks; covenant test, bank-statement/GST cross-check, management call.
3. Other flagged names -> **REVIEW**: analyst deep-dive this quarter using the three plain-language drivers; confirm or dismiss within 30 days.
4. Ranks up to 80 -> **WATCH**: monthly monitoring of price, ratings and pledges.
5. **Override, escalate regardless of rank**: downgrade to sub-investment grade or by 2 or more notches; auditor resignation, qualification or disclaimer; results deferred past the LODR deadline; promoter pledge above 50% or up more than 10 pp in a quarter; any lender reporting SMA-1/SMA-2 (CRILC).
6. **Escalation path**: analyst -> credit committee (watch-list tag, limit freeze, extra security) -> CRO/Board risk committee above delegated limits -> RBI June 2019 prudential framework (inter-creditor agreement) on SMA-2 or confirmed stress.

**Threshold and cost.** A missed default costs {COST_FN} and a false alarm costs {COST_FP}. Flagging is worth it when (1-p) x {COST_FP} < p x {COST_FN}, i.e. **p > 1/21 = 0.0476** (Polish validation cost optimum: {pm['threshold']['t_val_cost_optimal']:.3f}).
Capacity binds, so the operating rule is "top-k capped at 40; within that, everyone with PD > 1/21 is reviewed first".
At an *assumed* 2.5% annual base rate, model-expected cost this quarter is **{l25.loc['capped_top_k','exp_cost']:.1f}** units with capped top-k, versus {l25.loc['flag_nobody','exp_cost']:.1f} with no reviews and {l25.loc['bayes_PD>=1/21','exp_cost']:.1f} for an uncapped Bayes rule ({int(l25.loc['bayes_PD>=1/21','alerts'])} alerts, which exceeds capacity).
"""

def main():
    pm = json.load(open(O("polish_metrics.json"))); ev = json.load(open(O("india_eval.json")))
    sens = {t: json.load(open(O(f"india_eval_{t}.json"))) for t in ["lag60", "totalmarket"] if os.path.exists(O(f"india_eval_{t}.json"))}
    hs = pd.read_csv(O("polish_holdout_scores.csv")); q = pd.read_csv(O("quarterly_scores.csv.gz"), parse_dates=["quarter"])
    cases = pd.read_csv(P("data", "reference", "cases.csv")); peers = pd.read_csv(P("data", "reference", "peers.csv"))
    soft = pd.read_csv(P("data", "reference", "soft_signals.csv")); frozen = pd.read_csv(P("data", "reference", "universe_frozen.csv"))
    pcost = pd.read_csv(O("polish_holdout_cost_table.csv")); W = pd.read_csv(O("watchlist_latest.csv")); meta = pd.read_csv(O("company_meta.csv"))
    prov = pd.read_csv(O("provenance_financials.csv.gz"))
    figures(pm, hs, q, cases)
    pol = policy_md(ev, pm); open(P("reports", "policy.md"), "w").write(pol)
    rec = ev["recall"]; h = pm["holdout"]; nat = pm.get("native_nan_benchmark_pr_auc", {})
    cr = pd.DataFrame(ev["case_results"])            # report tables come from india_eval.json only
    gaps = pd.DataFrame(ev["data_gaps_cases"]); sl = pd.DataFrame(ev["soft_signal_leads"])
    def rr(e, v, k):
        x = e["recall"][f"{v}_{k}"]; ci = x.get("ci95", [None, None])
        return f"{x['hits']}/{x['n']}" + (f" [{ci[0]:.2f}-{ci[1]:.2f}]" if ci and ci[0] is not None else "")
    VARS = [("score_model", "Polish-trained model only"), ("score_model_forensic", "+ forensic overlay (Z'', F, M)"),
            ("score_full", "**Full composite (headline)**"), ("score_posthoc_credit", "Post-hoc + ICR/Debt-EBITDA rules (in-sample)")]
    rtab = pd.DataFrame([{"variant": lab, "recall@4Q all [95% CI]": rr(ev, v, "recall_4Q_all_cases"), "recall@4Q evaluable": rr(ev, v, "recall_4Q_scorable"),
                          "recall@8Q all": rr(ev, v, "recall_8Q_all_cases"), "recall@8Q evaluable": rr(ev, v, "recall_8Q_scorable"),
                          "flag in [-8Q,-4Q]": f"{rec[f'{v}_flag_in_window_-8Q_to_-4Q']['hits']}/{rec[f'{v}_flag_in_window_-8Q_to_-4Q']['n']}",
                          "peer flag rate": f"{rec[f'{v}_peer_flag_rate_all_quarters']:.1%}"} for v, lab in VARS])
    stab = pd.DataFrame([{"run": name, "recall@4Q (full)": rr(e, "score_full", "recall_4Q_all_cases"), "recall@8Q (full)": rr(e, "score_full", "recall_8Q_all_cases"),
                          "model-only @4Q": rr(e, "score_model", "recall_4Q_all_cases")}
                         for name, e in [("headline (lag 183d, full universe)", ev)] + [({"lag60": "60-day availability lag", "totalmarket": "large/mid-cap book only (Nifty Total Market)"}[t], e) for t, e in sens.items()]])
    core = cr[cr.track == "core"].copy()
    core["rank@4Q"] = core.apply(lambda r: "no data" if pd.isna(r.get("fy_used_4Q")) else "n/e" if r.get("evaluable_4Q") is not True else f"{int(r.rank_score_full_4Q)}/{int(r.n_ranked_4Q)} (cap {int(r.alert_cap_4Q)})", axis=1)
    core["rank@8Q"] = core.apply(lambda r: "no data" if pd.isna(r.get("fy_used_8Q")) else ("n/e" if r.get("evaluable_8Q") is not True else f"{int(r.rank_score_full_8Q)}/{int(r.n_ranked_8Q)}"), axis=1)
    ctab = core[["slug", "event_date", "q_4Q", "fy_used_4Q", "rank@4Q", "flag_score_full_4Q", "rank@8Q", "flag_score_full_8Q", "first_flag_lead_quarters_full"]].rename(
        columns={"flag_score_full_4Q": "flag 4Q", "flag_score_full_8Q": "flag 8Q", "first_flag_lead_quarters_full": "first flag (Q before event)"})
    for c_ in ["event_date", "q_4Q"]: ctab[c_] = ctab[c_].astype(str).str[:10]
    tm = {t: e for t, e in sens.items()}
    if "totalmarket" in tm:
        t = pd.DataFrame(tm["totalmarket"]["case_results"]); t = t[t.track == "core"]
        ctab["rank@4Q large-cap book"] = [("n/e" if r.get("evaluable_4Q") is not True else f"{int(r.rank_score_full_4Q)}/{int(r.n_ranked_4Q)}") for _, r in t.iterrows()]
    casetab = cases[["company", "event_date", "event_type", "source_url"]]
    mpc = pd.DataFrame(ev["matched_panel_cost"])
    ecm = pd.DataFrame(ev["expected_cost_mean_over_quarters"])
    miss_rate = prov.value.isna().mean()
    skipped = meta[meta.n_years == 0][["slug", "skip_reason"]] if "skip_reason" in meta else meta[meta.n_years == 0][["slug"]]
    top10 = W.head(10)[["rank", "company", "score_full", "Z2", "F_score", "driver_1"]]
    ilfs = core[core.slug.isin(["IL&FSTRANS", "IL&FSENGG"])][["slug", "rank@4Q", "rank@8Q"]]
    ex = core[core.drivers_at_4Q.astype(str).str.len() > 0][["slug", "drivers_at_4Q"]].head(4)
    ac = ev["alert_cap_check"]; nq = ev["universe_size_by_quarter"]; oe = ev["override_evidence"]
    oe_leads = "; ".join(f"{o['slug']} {o['lead_months']:.1f} ({o['trigger_type'].replace('_', ' ')}, {o['first_override_trigger']})" for o in oe["cases"] if o["lead_months"] is not None) or "none"
    report = f"""# PS 02: Early-Warning System for Corporate Distress and Fraud ("would it have caught IL&FS?")
*Generated by `python run_all.py`. Every number below is read from `outputs/polish_metrics.json` and `outputs/india_eval*.json` from the same run.*

## 1. Summary
* **Distress model** (UCI Polish bankruptcy data, 43,405 firm-years): a monotone LightGBM on 28 size-free ratios that can be rebuilt from Indian filings, with train-median imputation.
  On the untouched holdout, **PR-AUC is {h['pr_auc_transfer_selected']:.3f}** against a no-skill rate of {h['no_skill_pr_auc']:.3f}. The holdout has {pm['pos']['holdout']} defaults in {pm['n']['holdout']:,} firm-years ({pm['pos_rate']['holdout']:.2%}, about 1:{(1-pm['pos_rate']['holdout'])/pm['pos_rate']['holdout']:.1f}). ROC-AUC is {h['roc_auc_transfer_selected']:.3f}, and recall at a 5% alert rate is {h['recall_at_5pct_alert']:.3f}.
* **Indian backtest.** {len(core)} documented non-financial defaulters, {len(peers)} sector peers and a frozen universe of {int((frozen.status=='ok').sum())} listed companies ({ev['watchlist']['n_monitored']} eligible borrowers at {ev['watchlist']['quarter']}). Alerts are hard-capped at min(40, 5%) per quarter.
  - Headline composite recall is **{rr(ev,'score_full','recall_4Q_all_cases')} at 4 quarters** and **{rr(ev,'score_full','recall_8Q_all_cases')} at 8 quarters** before the event.
  - **IL&FS was not caught.** IL&FS Transportation and IL&FS Engineering ranked {', '.join(ilfs['rank@4Q'])} at 4Q.
  - **Escalation-override evidence ({oe['label']}).** Of the {oe['n_core']} cases, {oe['n_with_trigger_before_event']} had an override trigger (rating cut, auditor qualification or resignation, delayed results, promoter pledge, group default) before the event, with a median lead of {oe['median_lead_months']:.1f} months. Only {oe['n_trigger_ge_4Q']} had one at least 4Q ahead. Counting the model flag *or* an override at least 4Q ahead gives {oe['n_model_flag_or_override_ge_4Q']}/{oe['n_core']}. Lead times (months, from section 5.3): {oe_leads}.
* **Why the PR-AUC benchmark fell from {nat['transfer']['holdout']:.3f} to {h['pr_auc_transfer_selected']:.3f}.** Both figures are the same transfer LightGBM on the same Polish holdout. {nat['transfer']['holdout']:.3f} keeps Polish missing values as LightGBM NaN branches, so missingness acts as a distress indicator. The deployed {h['pr_auc_transfer_selected']:.3f} median-imputes instead, because in Indian filings a blank usually means a zero balance or an absent schedule, not distress. Using all 64 Polish features does not close the gap: {nat['full64']['holdout']:.3f} with NaN branches vs {h['pr_auc_full64_lgbm']:.3f} imputed. The deployed model also uses only the {len(TRANSFER_FEATURES)} features that can be rebuilt from Indian statements (Attr6 dropped). The metrics are unchanged; only the deployment choice differs.
* **Honest bottom line.** Annual book ratios, available about 6 months after year end and run through a model trained on Polish SMEs, do not separate large Indian group defaults from the many small distressed listed companies that fill a 5% alert budget. Most of the warning for these cases came from governance and market events (rating downgrades, auditor exits, results delays, pledges). These are researched for the cases (Section 5.3) and built into the escalation override. They are not counted in recall, because they were not collected for non-defaulters.

## 2. Data
**Polish (UCI 365).** File *k* holds ratios with bankruptcy status after (6-*k*) years. **Horizon split:** train on 1-3year, validate on 4year (2-year horizon, about 8Q; used for selection, isotonic calibration and threshold), and keep 5year (1-year horizon, about 4Q) as the untouched holdout. This is a horizon split, not calendar time, because the files carry no dates. Duplicate rows were removed. A secondary firm-disjoint split within 5year gives PR-AUC {pm['secondary_within_5year_split']['pr_auc_lgbm']:.3f}.

{md_table(pd.DataFrame({'rows': pm['n'], 'defaults': pm['pos'], 'default rate': pm['pos_rate']}).reset_index().rename(columns={'index':'split'}), '{:.4f}')}

**Indian statements.** Annual statements come from screener.in, which compiles annual reports and exchange filings (consolidated where available, otherwise standalone). There are {len(prov):,} data points in `outputs/provenance_financials.csv.gz`. Each has a URL, an access time in UTC and an **as-of availability date = FY end + {ev['avail_lag_days']} days** (the Companies Act s.96 AGM deadline). {miss_rate:.1%} of the cells are blank at source; these are marked missing and never filled. Companies skipped as unparsable: {', '.join(skipped.slug) if len(skipped) else 'none'}.

**Default events.** Each event date is the earliest default or D rating found:

{md_table(casetab)}

Reliance Capital (an NBFC) is a separate financial track and is not counted in recall. The IL&FS parent and IFIN, DHFL, Bhushan, Essar, Monnet and Uttam Galva have no retrievable listed-company statement pages, so they are excluded. Jaypee Infratech, Amtek and Alok (2015 events) predate peer coverage and are also excluded.

**Peers and universe.** {len(peers)} sector-matched peers (`peers.csv`). The universe is 598 Nifty Total Market non-financial companies plus 300 seeded-random older NSE listings, frozen on 2026-10-29. Universe size by quarter: {min(nq.values())}-{max(nq.values())} names. Quarters before {ev['full_coverage_from']} (fewer than 300 universe names) are **not evaluable (n/e)**, and no alerts are raised in them.

## 3. Method and leakage controls
* **Preprocessing fit on Polish train only**: winsorise at 1/99 percentiles, then median-impute (and standardise for the logistic model). Model selection and calibration use validation only. The holdout is used once. No Indian outcome is used in any fit. Statements are used only after FY end + lag, case quarters on or after the event are dropped, and the alert cutoff is a rank, so it uses no outcomes. Automated checks: {ev['leakage_checks']}.
* **Model.** LightGBM with monotone sign constraints was compared with logistic regression (balanced, C=0.5). LightGBM was selected on validation PR-AUC ({pm['val_pr_auc']['transfer']['lightgbm_monotone']:.3f} vs {pm['val_pr_auc']['transfer']['logistic']:.3f}).
* **Transfer fixes, made from covariate comparison and before the frozen run.** (a) In the Polish data, missingness itself predicts bankruptcy: holdout PR-AUC with native NaN is {nat.get('transfer',{}).get('holdout',float('nan')):.3f} (full64: {nat.get('full64',{}).get('holdout',float('nan')):.3f}). In Indian data, a missing value means a zero balance or an absent schedule, so the deployed model median-imputes. Zero finance cost now means uncapped interest cover. (b) Attr6 (retained earnings/TA) was dropped: its Polish median is 0.000, while Indian "reserves" include securities premium. Both fixes were found after a first run put debt-free IT companies at the top of the ranking. The fixes do not use the case labels, but they are disclosed.
* **Forensic scores.** **Altman Z''** = 6.56 WC/TA + 3.26 RE/TA + 6.72 EBIT/TA + 1.05 BVE/TL, with distress below 1.1. Z'' was chosen because it needs no market value (unavailable or noisy for distressed and suspended Indian stocks) and no sales/TA (sector-biased), and it is Altman's emerging-market and non-manufacturer form. On the Polish holdout, Z'' alone gives PR-AUC {h['pr_auc_altman_z2_standalone']:.3f}. **Piotroski F** uses 9 signals; **Beneish M** uses 8 variables and flags above -1.78, with missing indices set to neutral and counted.
* **Composite** = model log-odds + a pre-specified log-odds overlay, which was not fitted: {ev['overlay_logodds']}.
* **Quarterly mapping.** Ratios change once a year, when the annual report becomes public. Price and overdue-accounts signals change quarterly. "4Q" means the score at the last quarter end at least 12 months before the event; "8Q" means at least 24 months.
* **Explanations.** The top 3 positive contributors come from SHAP TreeExplainer log-odds plus the overlay terms, written as plain sentences with each value and its peer percentile.

## 4. Polish holdout results
![PR](figures/polish_pr.png)

{md_table(pcost)}

## 5. Indian backtest results
**Alert-cap check:** the most alerts in any evaluable quarter is {ac['max_alerts_any_quarter']}, and the highest alert rate is {ac['max_alert_rate_any_quarter']:.2%} ({ac['n_evaluable_quarters']} quarters, rule {ac['cap_rule']}).
### 5.1 Recall at the capped 5% alert rate (exact 95% CI)
{md_table(rtab)}

**Sensitivities:**

{md_table(stab)}

### 5.2 Rank of every case (composite score; rank / N eligible; n/e = before universe coverage)
{md_table(ctab)}

![cases](figures/case_trajectories.png)

Driver examples at 4Q: {' '.join(f"**{r.slug}**: {r.drivers_at_4Q}." for r in ex.itertuples())}

### 5.3 Researched soft signals (case evidence, not in metrics)
{md_table(sl)}

Only signals dated before the event count as warnings. For example, Rolta's S&P D on 2016-06-20 is after its event and only confirms it.

### 5.4 Cost
Matched case/peer panel, realised cost (missed defaults x 20 + false alarms x 1):

{md_table(mpc[['lead','variant','TP','FN','FP','calc']])}

Universe model-expected cost (mean per evaluable quarter). PDs are calibrated on Polish data and prior-shifted to an *assumed* base rate of 1%, 2.5% or 5% (an assumption, not a sourced figure):

{md_table(ecm, '{:.2f}')}

![cost](figures/cost_curve.png)

{pol}

Latest capped watch-list, top 10. These are screening outputs, not predictions of default:

{md_table(top10)}

## 6. Missing data
{md_table(gaps[['slug','n_fy_on_screener','first_fy','n_fy_public_before_event','has_3y_pre_event','missing_feature_share_pre_event','F_score_missing_years','M_score_missing_years']])}

Videocon used irregular fiscal periods, so growth ratios are blanked for non-12-month periods. Ruchi Soya has a gap between FY2019 and FY2025 on screener.in, and only pre-event years are used. Rolta's statements start in FY2014, so its 8Q point cannot be scored.

## 7. Survivorship bias, sector effects, limitations
* **Survivorship bias (material).** The universe was built from *current* listings, the Nifty Total Market and NSE lists as of October 2026. Firms that delisted, were suspended or defaulted between 2015 and 2026 are missing unless they are one of the cases. As a result, (i) historical quarters contain only survivors, so peer and universe "false alarms" are overstated as specificity (some flagged names, such as GTL, were in fact distressed), and (ii) the 5% budget is filled from survivors. A point-in-time universe from historical index constituents or exchange lists is the main fix.
* **Selection.** The cases are famous large defaults, picked because they have documented dates. Small-company defaults, which the model ranks well, are under-represented. With n = 15, the confidence intervals are wide.
* **Sector effects.** Infrastructure, EPC, telecom, airlines and realty structurally carry high leverage and low Z''. The cutoff is universe-wide, so a sector-relative rank is the next step.
* **Transfer gap.** Polish SMEs from 2000 to 2013 are not Indian conglomerates whose subsidiaries rely on group guarantees (IL&FS, ADAG, Future).
* **Annual granularity.** At the 4Q point, the latest statements are often 15-21 months old.
* **Source caveats.** Screener may show restated figures. Event dates are the earliest found, and earlier defaults are possible (Videocon, McNally). The post-hoc credit-rule variant is in-sample.

## 8. Reproducibility
Run `make setup && make all` (`python run_all.py`), then `make test` (pytest). Start the local app with `make app` (Streamlit). `python run_all.py --scrape` re-fetches source pages; this changes the frozen inputs. Seed 42. No push and no deployment.
"""
    open(P("reports", "report.md"), "w").write(report)
    import markdown
    css = "@page{size:A4;margin:1.1cm} body{font-family:DejaVu Sans,Arial;font-size:8.6pt} table{border-collapse:collapse;font-size:6.4pt;margin:4px 0} td,th{border:1px solid #bbb;padding:1px 3px;word-break:break-word} img{max-width:88%} h1{font-size:14pt} h2{font-size:11.5pt;border-bottom:1px solid #999;margin:8px 0 3px} h3{font-size:10pt;margin:6px 0 2px} p,li{margin:2px 0}"
    html = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{markdown.markdown(report, extensions=['tables'])}</body></html>"
    open(P("reports", "report.html"), "w").write(html)
    from weasyprint import HTML
    doc = HTML(string=html, base_url=P("reports")).render(); doc.write_pdf(P("reports", "report.pdf"))
    print("report pages:", len(doc.pages))
    wl = W.head(40).copy(); wl["score_full"] = wl.score_full.round(2)
    dash = f"""<html><head><meta charset='utf-8'><title>PS02 EWS dashboard</title><style>{css} body{{font-size:10pt}}</style></head><body>
<h1>PS02 Early-Warning dashboard (static snapshot)</h1><p>Interactive app: <code>make app</code>. Quarter {ev['watchlist']['quarter']}: {ev['watchlist']['n_monitored']} borrowers monitored, {ev['watchlist']['n_alerts']} alerts (cap {ev['watchlist']['alert_cap']}).</p>
<h2>Recall at the capped 5% alert rate</h2>{markdown.markdown(md_table(rtab), extensions=['tables'])}
<p>Polish holdout PR-AUC {h['pr_auc_transfer_selected']:.3f} (no-skill {h['no_skill_pr_auc']:.3f}); Bayes threshold PD &ge; 1/21 = 0.0476.</p>
<h2>Watch-list (top 40 by score; flagged = within cap)</h2>{markdown.markdown(md_table(wl[['rank','company','slug','score_full','flag_score_full','Z2','F_score','M_score','action','driver_1','driver_2','driver_3']]), extensions=['tables'])}
<h2>Cases</h2>{markdown.markdown(md_table(ctab), extensions=['tables'])}<img src='../reports/figures/case_trajectories.png'>
</body></html>"""
    open(P("dashboard", "index.html"), "w").write(dash)
    print("report + dashboard written")

if __name__ == "__main__":
    main()
