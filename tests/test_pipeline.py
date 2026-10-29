"""Tests run against the committed/regenerated outputs of `python run_all.py` plus pure unit tests."""
import json, math, pathlib, re
import numpy as np, pandas as pd, pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"; REF = ROOT / "data" / "reference"
CAP = 40; RATE = 0.05

@pytest.fixture(scope="module")
def q():
    return pd.read_csv(OUT / "quarterly_scores.csv.gz", parse_dates=["quarter", "available_from", "fy_end"])

@pytest.fixture(scope="module")
def ev():
    return json.load(open(OUT / "india_eval.json"))

@pytest.fixture(scope="module")
def cases():
    return pd.read_csv(REF / "cases.csv", parse_dates=["event_date"])

# ---------- alert cap ----------
@pytest.mark.parametrize("variant", ["score_full", "score_model", "score_model_forensic", "score_posthoc_credit"])
def test_no_quarter_exceeds_alert_cap(q, variant):
    e = q[q.eligible]
    g = e.groupby("quarter").agg(n=("slug", "size"), alerts=("flag_" + variant, "sum"))
    cap = np.minimum(CAP, np.floor(RATE * g.n))
    bad = g[g.alerts > cap]
    assert bad.empty, f"quarters over cap: {bad}"
    assert (g.alerts <= CAP).all()

def test_cap_is_used_in_evaluable_quarters(q):
    e = q[q.eligible & q.evaluable_quarter]
    g = e.groupby("quarter").agg(n=("slug", "size"), alerts=("flag_score_full", "sum"))
    assert (g.alerts == np.minimum(CAP, np.floor(RATE * g.n))).all()
    assert q[~q.evaluable_quarter].flag_score_full.sum() == 0

def test_flags_are_top_ranks(q):
    e = q[q.eligible & q.flag_score_full]
    assert (e.rank_score_full <= e.alert_cap).all()

# ---------- leakage ----------
def test_features_public_before_quarter(q):
    assert (q.available_from <= q.quarter).all()
    lag = (q.available_from - q.fy_end).dt.days
    assert (lag >= 60).all()

def test_no_case_quarter_on_or_after_event(q, cases):
    m = q.merge(cases[["slug", "event_date"]], on="slug")
    assert (m.quarter < m.event_date).all()

def test_case_4q_8q_points_precede_event(ev):
    for r in ev["case_results"]:
        ed = pd.Timestamp(r["event_date"])
        for L, months in [("4Q", 12), ("8Q", 24)]:
            if r.get(f"q_{L}"):
                assert pd.Timestamp(r[f"q_{L}"]) <= ed - pd.DateOffset(months=months)
            if r.get(f"fy_used_{L}") and r.get(f"available_from_{L}"):
                assert pd.Timestamp(r[f"available_from_{L}"]) <= pd.Timestamp(r[f"q_{L}"])

def test_leakage_flags_true(ev):
    lc = ev["leakage_checks"]
    assert lc["feature_rows_available_before_quarter"] and lc["no_case_quarter_on_or_after_event"]

def test_soft_signals_counted_as_4q_precede_event_minus_12m(ev, cases):
    soft = pd.read_csv(REF / "soft_signals.csv", parse_dates=["signal_date"])
    ce = cases.set_index("slug").event_date
    for r in ev["soft_signal_leads"]:
        ss = soft[soft.slug == r["slug"]]
        assert r["n_at_least_4Q_before"] == int((ss.signal_date <= ce[r["slug"]] - pd.DateOffset(months=12)).sum())
        if r["earliest_signal"]:
            assert pd.Timestamp(r["earliest_signal"]) < ce[r["slug"]]
    # Rolta's S&P 'D' (2016-06-20) is after its 2016-05-16 event: never a warning
    rol = next(r for r in ev["soft_signal_leads"] if r["slug"] == "ROLTA")
    assert rol["n_before_event"] == 0 or pd.Timestamp(rol["earliest_signal"]) < pd.Timestamp("2016-05-16")

def test_soft_signals_not_used_as_model_features():
    src = (ROOT / "src" / "score_india.py").read_text()
    main_part = src.split("def main")[1].split("soft_signals.csv")[0]
    assert "flag_" not in "".join(l for l in main_part.splitlines() if "soft" in l)

# ---------- cost ----------
def test_bayes_threshold():
    from common import COST_FN, COST_FP
    assert COST_FN == 20 and COST_FP == 1
    assert math.isclose(COST_FP / (COST_FP + COST_FN), 1 / 21)

def test_expected_cost_curve_arithmetic():
    from polish_model import expected_cost_curve
    y = np.array([1, 1, 0, 0, 0, 0]); p = np.array([.9, .02, .5, .01, .3, .02])
    r = expected_cost_curve(y, p, [0.0476]).iloc[0]
    assert (r.TP, r.FN, r.FP) == (1, 1, 2) and r.cost == 20 * 1 + 1 * 2

def test_matched_panel_cost_strings(ev):
    for r in ev["matched_panel_cost"]:
        assert r["cost_units"] == 20 * r["FN"] + r["FP"]
        assert str(r["cost_units"]) in r["calc"]

def test_polish_threshold(ev):
    pm = json.load(open(OUT / "polish_metrics.json"))
    assert math.isclose(pm["threshold"]["t_bayes"], 1 / 21, rel_tol=1e-6)

# ---------- robustness ----------
def test_md_table_handles_float_none_nan():
    from make_report import md_table
    df = pd.DataFrame({"a": [1.23456, None, np.nan], "b": [None, "x|y", 3], "c": [True, False, None]})
    s = md_table(df)
    assert s.count("\n") == 5 and "1.235" in s and "nan" not in s and "None" not in s

def test_load_company_skips_empty(tmp_path):
    from india_features import load_company
    p = tmp_path / "X.json"; p.write_text(json.dumps({"slug": "X", "error": "no tables"}))
    df, prov, d = load_company(p)
    assert df is None
    p.write_text(json.dumps({"slug": "X", "tables": {}, "fetches": []}))
    df, prov, d = load_company(p)
    assert df is None or df.empty

def test_skipped_companies_logged():
    meta = pd.read_csv(OUT / "company_meta.csv")
    sk = meta[meta.n_years == 0]
    assert "skip_reason" in meta and sk.skip_reason.notna().all()

# ---------- frozen universe & report consistency ----------
def test_frozen_universe(q):
    f = pd.read_csv(REF / "universe_frozen.csv")
    assert f.frozen_as_of.nunique() == 1 and f.slug.is_unique
    uni = q[q.group == "universe"].slug.unique()
    assert set(uni) <= set(f[f.status == "ok"].slug)

def test_report_matches_eval(ev):
    rep = (ROOT / "reports" / "report.md").read_text()
    r = ev["recall"]["score_full_recall_4Q_all_cases"]
    assert f"{r['hits']}/{r['n']}" in rep
    r8 = ev["recall"]["score_full_recall_8Q_all_cases"]
    assert f"{r8['hits']}/{r8['n']}" in rep
    for c in ev["case_results"]:
        if c["track"] == "core" and c.get("evaluable_4Q"):
            assert f"{int(c['rank_score_full_4Q'])}/{int(c['n_ranked_4Q'])}" in rep, c["slug"]
    pm = json.load(open(OUT / "polish_metrics.json"))
    assert f"{pm['holdout']['pr_auc_transfer_selected']:.3f}" in rep
    assert (ROOT / "reports" / "report.pdf").stat().st_size > 10000
    assert (ROOT / "dashboard" / "index.html").exists()

def test_cases_have_sources():
    c = pd.read_csv(REF / "cases.csv")
    assert (c.track == "core").sum() >= 10 and c.source_url.str.startswith("http").all()
    assert len(pd.read_csv(REF / "peers.csv")) >= 30
    s = pd.read_csv(REF / "soft_signals.csv")
    assert s.source_url.str.startswith("http").all()

# ---------- page 1 of the PDF ----------
def test_pdf_page1_has_recall_override_and_prauc_explanation(ev):
    from pypdf import PdfReader
    p1 = " ".join(PdfReader(str(ROOT / "reports" / "report.pdf")).pages[0].extract_text().split())
    r = ev["recall"]["score_full_recall_4Q_all_cases"]; r8 = ev["recall"]["score_full_recall_8Q_all_cases"]
    assert f"{r['hits']}/{r['n']}" in p1 and f"{r8['hits']}/{r8['n']}" in p1
    oe = ev["override_evidence"]
    assert "Escalation-override evidence" in p1 and "Lead times" in p1
    assert f"{oe['n_model_flag_or_override_ge_4Q']}/{oe['n_core']}" in p1
    for o in oe["cases"]:
        if o["lead_months"] is not None:
            assert f"{o['slug']} {o['lead_months']:.1f}" in p1, o["slug"]
    pm = json.load(open(OUT / "polish_metrics.json"))
    assert "Why the PR-AUC benchmark fell" in p1
    assert f"{pm['native_nan_benchmark_pr_auc']['transfer']['holdout']:.3f}" in p1
    assert f"{pm['holdout']['pr_auc_transfer_selected']:.3f}" in p1

def test_override_evidence_consistent(ev):
    soft = pd.read_csv(REF / "soft_signals.csv", parse_dates=["signal_date"])
    cases = pd.read_csv(REF / "cases.csv", parse_dates=["event_date"]).set_index("slug")
    for o in ev["override_evidence"]["cases"]:
        if o["first_override_trigger"]:
            d = pd.Timestamp(o["first_override_trigger"]); e = cases.event_date[o["slug"]]
            assert d < e and o["override_ge_4Q"] == (d <= e - pd.DateOffset(months=12))
            assert d in set(soft[soft.slug == o["slug"]].signal_date)
