"""Experimento CIMIS: línea base determinista frente a módulo híbrido cuantílico.

Uso:
  python run_experiment.py --tag base --f0 30 --snr 4 --f_lf 4
  python run_experiment.py --tag snr2 --snr 2 --no-tune --params results/best_params.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

import build_dataset as B
import cimis_core as C

RES = "/home/claude/cimis/results/"
TARGETS = ["PHIE", "VSH"]
QS = (0.1, 0.5, 0.9)
GRID = {"learning_rate": [0.05], "max_leaf_nodes": [7, 15],
        "min_samples_leaf": [100, 400]}
METHODS = ["REF", "LIN", "QGB_L", "QGB", "QGB_C"]


# --------------------------------------------------------------------------- #
def qgb(q, params, seed, mono=None):
    return HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_iter=200, l2_regularization=1.0,
        early_stopping=False, random_state=seed, monotonic_cst=mono, **params)


def mono_vec(feats, target):
    m = C.MONO.get(target, {})
    return [m.get(f, 0) for f in feats]


def fit_predict_hybrid(A, B, feats, target, params, seed, conformal=False):
    """Gradient boosting cuantílico (P10, P50, P90) con restricciones de monotonía.
    Si conformal=True, los cuantiles extremos se calibran con regresión cuantílica
    conformalizada (CQR) usando un 25 % de los pozos de entrenamiento como
    conjunto de calibración."""
    mono = mono_vec(feats, target)
    Xa, ya, Xb = A[feats].values, A[target].values, B[feats].values
    p50 = qgb(0.5, params, seed, mono).fit(Xa, ya).predict(Xb)
    if not conformal:
        lo = qgb(0.1, params, seed, mono).fit(Xa, ya).predict(Xb)
        hi = qgb(0.9, params, seed, mono).fit(Xa, ya).predict(Xb)
        return np.sort(np.column_stack([lo, p50, hi]), axis=1), 0.0
    wells = np.array(sorted(A.WELL.unique()))
    rng = np.random.default_rng(seed)
    cal = rng.choice(wells, size=max(2, int(round(0.25 * len(wells)))), replace=False)
    pr, ca = A[~A.WELL.isin(cal)], A[A.WELL.isin(cal)]
    mlo = qgb(0.1, params, seed, mono).fit(pr[feats].values, pr[target].values)
    mhi = qgb(0.9, params, seed, mono).fit(pr[feats].values, pr[target].values)
    yc = ca[target].values
    E = np.maximum(mlo.predict(ca[feats].values) - yc, yc - mhi.predict(ca[feats].values))
    n = len(E)
    Q = float(np.quantile(E, min(1.0, np.ceil((n + 1) * 0.8) / n)))
    lo, hi = mlo.predict(Xb) - Q, mhi.predict(Xb) + Q
    return np.sort(np.column_stack([lo, p50, hi]), axis=1), Q


def fit_predict_linear(A, B, feats, target):
    """Transformación determinista calibrada por mínimos cuadrados; intervalos a
    partir de los cuantiles empíricos de los residuos de entrenamiento."""
    m = LinearRegression().fit(A[feats].values, A[target].values)
    res = A[target].values - m.predict(A[feats].values)
    q10, q90 = np.quantile(res, [0.1, 0.9])
    p50 = m.predict(B[feats].values)
    return np.column_stack([p50 + q10, p50, p50 + q90]), m


def tune(df, target, feats, seed):
    """Búsqueda en rejilla con validación agrupada por pozo (3 particiones)
    sobre el modelo P50; criterio: pérdida de pinball (equivale a MAE/2)."""
    gkf = GroupKFold(n_splits=3)
    X, y, g = df[feats].values, df[target].values, df.WELL.values
    mono = mono_vec(feats, target)
    best, best_loss = None, np.inf
    for lr, mln, msl in product(*GRID.values()):
        p = {"learning_rate": lr, "max_leaf_nodes": mln, "min_samples_leaf": msl}
        loss = []
        for tr, te in gkf.split(X, y, g):
            pred = qgb(0.5, p, seed, mono).fit(X[tr], y[tr]).predict(X[te])
            loss.append(np.mean(np.abs(y[te] - pred)))
        if np.mean(loss) < best_loss:
            best, best_loss = p, float(np.mean(loss))
    return best, best_loss


# --------------------------------------------------------------------------- #
def metrics(y, P):
    p50 = P[:, 1]
    e = p50 - y
    cov = np.mean((y >= P[:, 0]) & (y <= P[:, 2]))
    width = np.mean(P[:, 2] - P[:, 0])
    # puntuación de intervalo (Gneiting y Raftery, 2007) para alfa = 0,2
    a = 0.2
    isc = np.mean((P[:, 2] - P[:, 0]) + (2 / a) * (P[:, 0] - y) * (y < P[:, 0])
                  + (2 / a) * (y - P[:, 2]) * (y > P[:, 2]))
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2)),
            "bias": float(np.mean(e)), "r": float(np.corrcoef(y, p50)[0, 1]),
            "cov80": float(cov), "width80": float(width), "IS80": float(isc)}


def per_well(df, pcols):
    out = {}
    for w, g in df.groupby("WELL"):
        y = g["y"].values
        out[w] = {k: float(np.sqrt(np.mean((g[k + "_p50"].values - y) ** 2))) for k in pcols}
    return pd.DataFrame(out).T


def paired_tests(pw, a, b, n_boot=2000, seed=0):
    d = pw[a].values - pw[b].values          # >0 significa que b es mejor
    sw = stats.shapiro(d)
    if sw.pvalue >= 0.05:
        t = stats.ttest_rel(pw[a], pw[b], alternative="greater")
        test, p = "t pareada", t.pvalue
    else:
        t = stats.wilcoxon(pw[a], pw[b], alternative="greater")
        test, p = "Wilcoxon", t.pvalue
    rng = np.random.default_rng(seed)
    boots = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n_wells": int(len(d)), "shapiro_p": float(sw.pvalue), "test": test,
            "p": float(p), "mean_diff": float(np.mean(d)), "ci95": [float(lo), float(hi)],
            "frac_wells_improved": float(np.mean(d > 0))}


# --------------------------------------------------------------------------- #
def predict_all(A, B, target, mode, params, seed):
    F, L, LR_ = C.FEATURES[mode], C.LINEAR[mode], C.LINEAR_REF[mode]
    out = {}
    out["REF"], _ = fit_predict_linear(A, B, LR_, target)
    out["LIN"], _ = fit_predict_linear(A, B, L, target)
    if mode == "elastic":
        out["LIN_A"], _ = fit_predict_linear(A, B, ["lnAI_inv"], target)
    out["QGB_L"], _ = fit_predict_hybrid(A, B, L, target, params, seed)
    out["QGB"], _ = fit_predict_hybrid(A, B, F, target, params, seed)
    Pc, Q = fit_predict_hybrid(A, B, F, target, params, seed, conformal=True)
    Pc[:, 1] = out["QGB"][:, 1]
    out["QGB_C"] = np.sort(Pc, axis=1)
    return out, Q


def auc_sand_shale(df, p50):
    """AUC de discriminación arena/lutita (litología interpretada por FORCE)
    usando el Vsh P50 predicho; no interviene en el entrenamiento."""
    from sklearn.metrics import roc_auc_score
    m = (df.SAND.values > 0.5) | (df.SHALE.values > 0.5)
    if m.sum() < 50 or len(np.unique(df.SHALE.values[m] > 0.5)) < 2:
        return float("nan")
    return float(roc_auc_score(df.SHALE.values[m] > 0.5, p50[m]))


def run(cfg, tag, mode, tune_flag=True, params_file=None, seed=2026):
    t0 = time.time()
    wells = pickle.load(open(RES + f"wells_{mode}.pkl", "rb"))
    D = B.build(cfg, wells, mode)
    D = D[D.BAD < 0.5].reset_index(drop=True)          # control de calidad
    train = D[D.split == "train"].reset_index(drop=True)
    blind = D[D.split == "blind"].reset_index(drop=True)
    params = json.load(open(params_file)) if params_file else {}
    F = C.FEATURES[mode]
    ties = D.groupby("WELL")[["corr_tie", "corr_inv_true"]].first()
    summary = {"cfg": cfg, "mode": mode,
               "n_train_wells": int(train.WELL.nunique()),
               "n_blind_wells": int(blind.WELL.nunique()),
               "train_wells": sorted(train.WELL.unique().tolist()),
               "blind_wells": sorted(blind.WELL.unique().tolist()),
               "n_train_samples": int(len(train)), "n_blind_samples": int(len(blind)),
               "corr_len_ms": D.attrs.get("corr_len_ms"), "Sigma0": D.attrs.get("Sigma0"),
               "tie_median": float(ties.corr_tie.median()),
               "tie_range": [float(ties.corr_tie.min()), float(ties.corr_tie.max())],
               "inv_true_corr_median": float(ties.corr_inv_true.median()),
               "inv_true_corr_iqr": [float(ties.corr_inv_true.quantile(.25)),
                                     float(ties.corr_inv_true.quantile(.75))],
               "results": {}}
    preds_all = []
    for target in TARGETS:
        if tune_flag:
            params[target], _ = tune(train, target, F, seed)
        p = params[target]
        oof = {}
        for tr, te in GroupKFold(n_splits=5).split(train, groups=train.WELL):
            P, _ = predict_all(train.iloc[tr], train.iloc[te], target, mode, p, seed)
            for k in P:
                oof.setdefault(k, np.zeros((len(train), 3)))[te] = P[k]
        bp, Q = predict_all(train, blind, target, mode, p, seed)
        res = {"params": p, "cqr_Q": Q, "cv": {}, "blind": {}}
        for split, df, PP in (("cv", train, oof), ("blind", blind, bp)):
            yy = df[target].values
            for k, P in PP.items():
                res[split][k] = metrics(yy, P)
                if target == "VSH":
                    res[split][k]["AUC_sand_shale"] = auc_sand_shale(df, P[:, 1])
            tmp = pd.DataFrame({"WELL": df.WELL.values, "y": yy, "t": df.t.values,
                                "tvd": df.tvd.values,
                                "SAND": df.SAND.values, "SHALE": df.SHALE.values,
                                "lnAI_true": df.lnAI_true.values,
                                "lnAI_inv": df.lnAI_inv.values,
                                "lnAI_lf": df.lnAI_lf.values})
            for k, P in PP.items():
                tmp[k + "_p10"], tmp[k + "_p50"], tmp[k + "_p90"] = P.T
            tmp["target"], tmp["split"] = target, split
            preds_all.append(tmp)
            pw = per_well(tmp, list(PP))
            res[split]["per_well_rmse"] = pw.to_dict()
            res[split]["test_LIN_vs_QGB"] = paired_tests(pw, "LIN", "QGB")
            res[split]["test_REF_vs_LIN"] = paired_tests(pw, "LIN", "REF")
            if "LIN_A" in pw:
                res[split]["test_LINA_vs_LIN"] = paired_tests(pw, "LIN_A", "LIN")
        m50 = qgb(0.5, p, seed, mono_vec(F, target)).fit(train[F].values, train[target].values)
        pi = permutation_importance(m50, blind[F].values, blind[target].values,
                                    n_repeats=10, random_state=seed,
                                    scoring="neg_mean_absolute_error")
        res["perm_importance_blind"] = dict(zip(F, map(float, pi.importances_mean)))
        summary["results"][target] = res
    ps = [summary["results"][t]["cv"]["test_LIN_vs_QGB"]["p"] for t in TARGETS]
    adj = sorted(zip(ps, TARGETS))
    h1 = min(1.0, adj[0][0] * 2)
    h2 = max(h1, adj[1][0])
    summary["results"][adj[0][1]]["cv"]["test_LIN_vs_QGB"]["p_holm"] = h1
    summary["results"][adj[1][1]]["cv"]["test_LIN_vs_QGB"]["p_holm"] = h2
    summary["runtime_s"] = time.time() - t0
    json.dump(summary, open(RES + f"summary_{tag}.json", "w"), indent=1)
    if tune_flag:
        json.dump(params, open(RES + f"best_params_{mode}.json", "w"), indent=1)
    pd.concat(preds_all).to_parquet(RES + f"preds_{tag}.parquet")
    D.to_parquet(RES + f"data_{tag}.parquet")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="base")
    ap.add_argument("--mode", default="acoustic", choices=["acoustic", "elastic"])
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--f0", type=float, default=30)
    ap.add_argument("--snr", type=float, default=4)
    ap.add_argument("--f_lf", type=float, default=4)
    ap.add_argument("--phase_err", type=float, default=0)
    ap.add_argument("--f0_err", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-tune", action="store_true")
    ap.add_argument("--params", default=None)
    a = ap.parse_args()
    cfg = dict(f0=a.f0, snr=a.snr, f_lf=a.f_lf, phase_err=a.phase_err,
               f0_err=a.f0_err, seed=a.seed)
    if a.prepare:
        pickle.dump(B.prepare_wells(a.mode), open(RES + f"wells_{a.mode}.pkl", "wb"))
    s = run(cfg, a.tag, a.mode, tune_flag=not a.no_tune, params_file=a.params, seed=a.seed)
    print(a.tag, "wells", s["n_train_wells"], s["n_blind_wells"], "tie", round(s["tie_median"], 3),
          "inv_r", round(s["inv_true_corr_median"], 3))
    for t in TARGETS:
        r = s["results"][t]
        print(t, r["params"], "Q", round(r["cqr_Q"], 4))
        for sp in ("cv", "blind"):
            tab = pd.DataFrame({k: r[sp][k] for k in r[sp] if isinstance(r[sp][k], dict) and "RMSE" in r[sp][k]}).T.round(3)
            print(" ", sp); print(tab.to_string())
            print("  LIN vs QGB", {k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in r[sp]["test_LIN_vs_QGB"].items()})
    print("runtime", round(s["runtime_s"], 1))
