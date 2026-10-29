"""Benchmark distress model on UCI Polish Companies Bankruptcy (id 365).

Split (horizon/time-aware, nothing tuned on the holdout):
  train      = 1year,2year,3year files (ratios observed 5,4,3 years before the status date)
  validation = 4year file (2-year horizon ~ 8 quarters): model selection, calibration, threshold
  HOLDOUT    = 5year file (1-year horizon ~ 4 quarters): untouched until final scoring
Rows in validation/holdout that exactly duplicate a training row are dropped (leakage control).
Secondary check: firm-disjoint 70/30 split inside 5year only (each firm appears once per file).
All preprocessing (winsorisation bounds, medians, scaler) is fit on TRAIN only.
"""
import json, pickle
import arff, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from common import P, SEED, seed_all, FEATS, TRANSFER_FEATURES, COST_FN, COST_FP, ALERT_RATE, Prep

seed_all()

def load(k):
    d = arff.load(open(P("data", "raw", "uci", f"{k}year.arff")))
    df = pd.DataFrame(d["data"], columns=[a[0] for a in d["attributes"]])
    df["y"] = df.pop("class").astype(int); df["horizon_years"] = 6 - k
    for c in df.columns[:-2]: df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def z2(X):  # Altman Z'' on Polish attrs: WC/TA=Attr3, RE/TA=Attr6, EBIT/TA=Attr7, BVE/TL=Attr8
    return 6.56 * X.Attr3 + 3.26 * X.Attr6 + 6.72 * X.Attr7 + 1.05 * X.Attr8

def zprep(tr, X):  # winsorise/impute the four Z'' inputs with train statistics
    c = ["Attr3", "Attr6", "Attr7", "Attr8"]; lo, hi = tr[c].quantile(0.01), tr[c].quantile(0.99)
    Xc = X[c].clip(lo, hi, axis=1); return Xc.fillna(tr[c].clip(lo, hi, axis=1).median())

def pr_auc(y, p): return float(average_precision_score(y, p))

def expected_cost_curve(y, p, thresholds):
    rows = []
    for t in thresholds:
        f = p >= t; fn = int(((~f) & (y == 1)).sum()); fp = int((f & (y == 0)).sum())
        rows.append(dict(threshold=float(t), alerts=int(f.sum()), alert_rate=float(f.mean()), FN=fn, FP=fp,
                         TP=int((f & (y == 1)).sum()), cost=COST_FN * fn + COST_FP * fp))
    return pd.DataFrame(rows)

def fit_models(Xtr, ytr, feats, mono):
    prep = Prep().fit(Xtr[feats])
    lr = LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced", random_state=SEED).fit(prep.lin(Xtr[feats]), ytr)
    gb = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                            monotone_constraints=mono, random_state=SEED, verbose=-1, deterministic=True, force_row_wise=True, n_jobs=2)
    gb.fit(prep.tree(Xtr[feats]), ytr)
    return prep, lr, gb

def main():
    d = {k: load(k) for k in range(1, 6)}
    allfeats = [f"Attr{i}" for i in range(1, 65)]
    tr = pd.concat([d[1], d[2], d[3]], ignore_index=True)
    trkeys = set(map(tuple, tr[allfeats].round(6).fillna(-9e9).values))
    def dedup(df):
        m = np.array([tuple(r) not in trkeys for r in df[allfeats].round(6).fillna(-9e9).values]); return df[m].reset_index(drop=True), int((~m).sum())
    va, dva = dedup(d[4]); te, dte = dedup(d[5])
    res = {"split": "train=1-3year files, val=4year, holdout=5year", "dropped_dupes": {"val": dva, "holdout": dte},
           "n": {"train": len(tr), "val": len(va), "holdout": len(te)},
           "pos_rate": {"train": float(tr.y.mean()), "val": float(va.y.mean()), "holdout": float(te.y.mean())},
           "pos": {"train": int(tr.y.sum()), "val": int(va.y.sum()), "holdout": int(te.y.sum())}}
    out = {}
    for name, feats in [("transfer", FEATS), ("full64", allfeats)]:
        mono = [TRANSFER_FEATURES[f][1] if f in TRANSFER_FEATURES else 0 for f in feats]
        prep, lr, gb = fit_models(tr, tr.y, feats, mono)
        pv_lr = lr.predict_proba(prep.lin(va[feats]))[:, 1]; pv_gb = gb.predict_proba(prep.tree(va[feats]))[:, 1]
        out[name] = dict(prep=prep, lr=lr, gb=gb, feats=feats,
                         val={"logistic": pr_auc(va.y, pv_lr), "lightgbm_monotone": pr_auc(va.y, pv_gb)})
    # model selection on VALIDATION only
    sel = out["transfer"]; best = max(sel["val"], key=sel["val"].get)
    raw = lambda m, X: (m["lr"].predict_proba(m["prep"].lin(X[m["feats"]]))[:, 1] if best == "logistic"
                        else m["gb"].predict_proba(m["prep"].tree(X[m["feats"]]))[:, 1])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=0.999).fit(raw(sel, va), va.y)
    # threshold: theory + empirical on validation
    pv = iso.predict(raw(sel, va))
    t_bayes = COST_FP / (COST_FP + COST_FN)
    grid = np.unique(np.r_[np.linspace(0.005, 0.6, 120), t_bayes])
    cv = expected_cost_curve(va.y.values, pv, grid); t_emp = float(cv.loc[cv.cost.idxmin(), "threshold"])
    # ---- final: untouched holdout ----
    pt = iso.predict(raw(sel, te)); ph_raw = raw(sel, te)
    ct = expected_cost_curve(te.y.values, pt, [t_bayes, t_emp, float(np.quantile(pt, 1 - ALERT_RATE))])
    ct["rule"] = ["bayes_1_over_21", "val_cost_optimal", "top_5pct_capacity"]
    base = dict(threshold=None, alerts=0, FN=int(te.y.sum()), FP=0, cost=COST_FN * te.y.sum(), rule="flag_nobody")
    top5 = pt >= np.quantile(pt, 1 - ALERT_RATE)
    res.update({
        "selected_model": best, "val_pr_auc": {k: v["val"] for k, v in out.items()},
        "holdout": {
            "pr_auc_transfer_selected": pr_auc(te.y, ph_raw), "roc_auc_transfer_selected": float(roc_auc_score(te.y, ph_raw)),
            "pr_auc_transfer_logistic": pr_auc(te.y, sel["lr"].predict_proba(sel["prep"].lin(te[FEATS]))[:, 1]),
            "pr_auc_transfer_lgbm": pr_auc(te.y, sel["gb"].predict_proba(sel["prep"].tree(te[FEATS]))[:, 1]),
            "pr_auc_full64_lgbm": pr_auc(te.y, out["full64"]["gb"].predict_proba(out["full64"]["prep"].tree(te[allfeats]))[:, 1]),
            "pr_auc_full64_logistic": pr_auc(te.y, out["full64"]["lr"].predict_proba(out["full64"]["prep"].lin(te[allfeats]))[:, 1]),
            "no_skill_pr_auc": float(te.y.mean()),
            "pr_auc_altman_z2_standalone": pr_auc(te.y, -z2(zprep(tr, te))),
            "roc_auc_altman_z2_standalone": float(roc_auc_score(te.y, -z2(zprep(tr, te)))),
            "recall_altman_z2_distress_zone": float((z2(te)[te.y == 1] < 1.1).mean()),
            "share_flagged_altman_z2_distress_zone": float((z2(te) < 1.1).mean()),
            "brier_calibrated": float(brier_score_loss(te.y, pt)),
            "recall_at_5pct_alert": float(top5[te.y.values == 1].mean()), "precision_at_5pct_alert": float(te.y.values[top5].mean()),
        },
        "threshold": {"bayes_formula": "flag if PD >= C_FP/(C_FP+C_FN) = 1/(1+20) = 0.0476", "t_bayes": t_bayes, "t_val_cost_optimal": t_emp},
    })
    # secondary: firm-disjoint split inside 5year only
    a, b = train_test_split(d[5], test_size=0.3, stratify=d[5].y, random_state=SEED)
    mono = [TRANSFER_FEATURES[f][1] for f in FEATS]
    p2, l2, g2 = fit_models(a, a.y, FEATS, mono)
    res["secondary_within_5year_split"] = {"n_test": len(b), "pos_rate": float(b.y.mean()),
        "pr_auc_lgbm": pr_auc(b.y, g2.predict_proba(p2.tree(b[FEATS]))[:, 1]),
        "pr_auc_logistic": pr_auc(b.y, l2.predict_proba(p2.lin(b[FEATS]))[:, 1])}
    # in-domain benchmark: same LightGBM but missing values left as NaN (LightGBM learns a "missing" branch).
    # In the Polish data missingness itself is predictive of bankruptcy; that pattern does not transfer to Indian
    # filings (zero balances / absent schedules), so the deployed transfer model uses train-median imputation.
    nat = {}
    for name, feats in [("transfer", FEATS), ("full64", allfeats)]:
        mono = [TRANSFER_FEATURES[f][1] if f in TRANSFER_FEATURES else 0 for f in feats]
        pp = out[name]["prep"]
        g3 = lgb.LGBMClassifier(**{**out[name]["gb"].get_params(), "monotone_constraints": mono}).fit(pp.clip(tr[feats]), tr.y)
        nat[name] = {"val": pr_auc(va.y, g3.predict_proba(pp.clip(va[feats]))[:, 1]), "holdout": pr_auc(te.y, g3.predict_proba(pp.clip(te[feats]))[:, 1])}
    res["native_nan_benchmark_pr_auc"] = nat
    # per-horizon PR-AUC (validation = 2y horizon, holdout = 1y horizon)
    res["pr_auc_by_horizon"] = {"2y (val, 4year file)": sel["val"][best], "1y (holdout, 5year file)": res["holdout"]["pr_auc_transfer_selected"]}
    cv.to_csv(P("outputs", "polish_val_cost_curve.csv"), index=False)
    pd.concat([ct, pd.DataFrame([base])]).to_csv(P("outputs", "polish_holdout_cost_table.csv"), index=False)
    pd.DataFrame({"y": te.y, "pd": pt, "raw": ph_raw}).to_csv(P("outputs", "polish_holdout_scores.csv"), index=False)
    pickle.dump({"prep": sel["prep"], "lr": sel["lr"], "gb": sel["gb"], "iso": iso, "selected": best, "feats": FEATS,
                 "t_bayes": t_bayes, "t_emp": t_emp, "val_pos_rate": float(va.y.mean()), "train_medians": sel["prep"].med},
                open(P("outputs", "transfer_model.pkl"), "wb"))
    json.dump(res, open(P("outputs", "polish_metrics.json"), "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))

if __name__ == "__main__":
    main()
