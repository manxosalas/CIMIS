"""Construye la tabla muestra a muestra (1 ms) para pozos de entrenamiento y
pozos ciegos, bajo una configuración de modelado/inversión dada."""
from __future__ import annotations

import numpy as np
import pandas as pd

import cimis_core as C

DATA = "/home/claude/force/repo/lithology_competition/data/"
SOURCES = {
    "train": [(DATA + "train.csv", None)],
    "blind": [(DATA + "hidden_test.csv", None),
              (DATA + "leaderboard_test_features.csv", DATA + "leaderboard_test_target.csv")],
}
REQ = {"acoustic": ["DTC", "RHOB", "GR"], "elastic": ["DTC", "DTS", "RHOB", "GR"]}


def prepare_wells(mode: str) -> dict:
    out = {"train": {}, "blind": {}}
    for split, files in SOURCES.items():
        for path, lith in files:
            df = C.load_interval(path, lith)
            for well, w in df.groupby("WELL"):
                seg = C.longest_valid_segment(w, REQ[mode])
                if seg is None:
                    continue
                seg["BAD"] = C.qc_flags(seg)
                out[split][well] = C.petrophysics(seg)
    return out


def _wavelets(cfg):
    w_true = C.ricker(cfg["f0"])
    w_inv = C.ricker(cfg["f0"] * cfg.get("f0_err", 1.0), phase_deg=cfg.get("phase_err", 0.0))
    return w_true, w_inv


def build(cfg: dict, wells: dict, mode: str) -> pd.DataFrame:
    w_true, w_inv = _wavelets(cfg)
    T = {s: {k: C.to_time(v) for k, v in d.items()} for s, d in wells.items()}
    lf = lambda x: C.lowpass(x, cfg["f_lf"])
    # estadísticos a priori estimados SOLO con pozos de entrenamiento
    keys = ["lnAI"] if mode == "acoustic" else ["lnAI", "lnSI", "lnRHO"]
    R = {k: [g[k].values - lf(g[k].values) for g in T["train"].values()] for k in keys}
    corr_len = C.corr_length(R["lnAI"])
    Sigma0 = np.cov(np.vstack([np.concatenate(R[k]) for k in keys]))
    rng = np.random.default_rng(cfg["seed"])
    snr_eff = cfg["snr"] if np.isfinite(cfg["snr"]) else 50.0
    rows = []
    for split in ("train", "blind"):
        for well, g in T[split].items():
            x = g.lnAI.values
            x_lf = lf(x)
            att = {"lnAI_true": x, "lnAI_lf": x_lf}
            if mode == "acoustic":
                d, s = C.synth_acoustic(x, w_true, cfg["snr"], rng)
                xi, var = C.invert_acoustic(d, x_lf, w_inv, np.mean(s ** 2) / snr_eff ** 2,
                                            float(Sigma0), corr_len)
                att.update(C.seismic_attrs(d))
                att["lnAI_inv"], att["lnAI_sd"] = xi, np.sqrt(var)
                tie = np.corrcoef(d, C.forward_acoustic_lin(x, w_inv))[0, 1]
            else:
                si, rho = g.lnSI.values, g.lnRHO.values
                m_lf = np.vstack([x_lf, lf(si), lf(rho)])
                K = float(np.mean(np.exp(2 * (m_lf[1] - m_lf[0]))))
                Dg, Sg = C.synth_elastic(x, si, rho, w_true, cfg["snr"], rng)
                sig_n2 = [np.mean(s ** 2) / snr_eff ** 2 for s in Sg]
                xi3, var3 = C.invert_elastic(Dg, m_lf, w_inv, sig_n2, Sigma0, corr_len, K)
                stack = Dg.mean(axis=0)
                sa = C.seismic_attrs(stack)
                att.update({"envelope": sa["envelope"], "inst_freq": sa["inst_freq"],
                            "amp": sa["amp"], "cos_phase": sa["cos_phase"]})
                ai, sii = np.exp(xi3[0]) / 1e6, np.exp(xi3[1]) / 1e6
                att.update({"lnAI_inv": xi3[0], "lnSI_inv": xi3[1], "lnRHO_inv": xi3[2],
                            "lnAI_sd": np.sqrt(var3[0]), "lnSI_sd": np.sqrt(var3[1]),
                            "vpvs_inv": ai / sii, "lambdarho": ai ** 2 - 2 * sii ** 2,
                            "murho": sii ** 2, "vpvs_true": np.exp(x - si),
                            "lnSI_true": si})
                a, b, c = C.fatti_coefs(C.ANGLES[0], K)
                near_lin = C.fftconvolve(a * np.diff(x) + b * np.diff(si) + c * np.diff(rho),
                                       w_inv, mode="same")
                tie = np.corrcoef(Dg[0], near_lin)[0, 1]
            att["lnAI_rel"] = att["lnAI_inv"] - x_lf
            att["lnAI_std21"] = C.roll(att["lnAI_inv"], 21, "std")
            att["tvd"] = g.TVD.values
            df = pd.DataFrame(att)
            for c in ["PHIE", "VSH", "BAD", "SAND", "SHALE", "t"]:
                df[c] = g[c].values
            df["WELL"], df["split"] = well, split
            df["corr_tie"] = float(tie)
            df["corr_inv_true"] = float(np.corrcoef(att["lnAI_inv"] - x_lf, x - x_lf)[0, 1])
            rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out.attrs.update({"corr_len_ms": corr_len * C.DT * 1000,
                      "Sigma0": np.atleast_2d(Sigma0).tolist()})
    return out
