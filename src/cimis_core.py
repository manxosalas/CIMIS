"""
Núcleo del flujo CIMIS para el experimento reproducible con registros FORCE 2020.

Módulos implementados:
  1. Ingesta y control de calidad de registros (intervalo jurásico siliciclástico).
  2. Petrofísica determinista: volumen de arcilla (Larionov, rocas antiguas) y
     porosidad efectiva por densidad con corrección por arcilla.
  3. Conversión profundidad-tiempo a partir del sónico y remuestreo a 1 ms.
  4. Modelado convolucional: acústico (incidencia normal) y preapilado
     (aproximación de Aki-Richards en la forma de Fatti) con ruido limitado en banda.
  5. Inversión bayesiana linealizada (acústica y elástica), con covarianza a priori
     exponencial y covarianza de ruido limitada en banda.
  6. Atributos sísmicos y elásticos para el módulo de aprendizaje automático.

Unidades: profundidad en m, tiempo doble en s, Vp y Vs en m/s, densidad en kg/m3,
impedancias en kg m^-2 s^-1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import solve
from scipy.signal import butter, fftconvolve, filtfilt, hilbert

JURASSIC_GROUPS = ["VIKING GP.", "VESTLAND GP.", "DUNLIN GP.", "BAAT GP."]
USECOLS = ["WELL", "DEPTH_MD", "X_LOC", "Y_LOC", "Z_LOC", "GROUP", "CALI", "BS",
           "DRHO", "RHOB", "GR", "DTC", "DTS", "FORCE_2020_LITHOFACIES_LITHOLOGY"]
RHO_MA, RHO_FL = 2.65, 1.00        # g/cm3 (matriz cuarzosa; filtrado base agua)
FT2M = 0.3048
DT = 0.001                         # s
ANGLES = (5.0, 15.0, 25.0, 35.0)   # grados


# --------------------------------------------------------------------------- #
# 1. Ingesta y control de calidad
# --------------------------------------------------------------------------- #
def load_interval(path: str, lith_path: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";", usecols=lambda c: c in USECOLS)
    if lith_path is not None:
        df["FORCE_2020_LITHOFACIES_LITHOLOGY"] = pd.read_csv(
            lith_path, sep=";")["FORCE_2020_LITHOFACIES_LITHOLOGY"].values
    df = df[df.GROUP.isin(JURASSIC_GROUPS)].copy()
    df = df.sort_values(["WELL", "DEPTH_MD"]).reset_index(drop=True)
    df.loc[(df.DTC < 40) | (df.DTC > 200), "DTC"] = np.nan
    df.loc[(df.DTS < 60) | (df.DTS > 400), "DTS"] = np.nan
    df.loc[(df.RHOB < 1.80) | (df.RHOB > 3.00), "RHOB"] = np.nan
    df.loc[(df.GR < 0) | (df.GR > 300), "GR"] = np.nan
    return df


def longest_valid_segment(w: pd.DataFrame, req: list[str], max_gap_m: float = 2.0,
                          min_len_m: float = 100.0) -> pd.DataFrame | None:
    """Interpola huecos cortos (<= max_gap_m) y conserva el tramo continuo más
    largo con las curvas requeridas válidas."""
    w = w.copy()
    step = np.nanmedian(np.diff(w.DEPTH_MD.values))
    lim = max(1, int(round(max_gap_m / step)))
    for c in req:
        w[c] = w[c].interpolate(limit=lim, limit_area="inside")
    ok = w[req].notna().all(axis=1).values
    brk = np.r_[True, np.diff(w.DEPTH_MD.values) > 3 * step]
    seg_id = np.cumsum(brk | np.r_[True, ok[1:] != ok[:-1]])
    best, best_len = None, 0.0
    for s in np.unique(seg_id[ok]):
        m = (seg_id == s) & ok
        L = np.ptp(w.DEPTH_MD.values[m]) if m.sum() > 1 else 0.0
        if L > best_len:
            best, best_len = m, L
    if best is None or best_len < min_len_m:
        return None
    return w[best].reset_index(drop=True)


def qc_flags(w: pd.DataFrame) -> np.ndarray:
    """Hoyo en mala condición: |CALI-BS| > 2 pulgadas o |DRHO| > 0,15 g/cm3."""
    bad = np.zeros(len(w), dtype=bool)
    if "BS" in w and w.BS.notna().any():
        bad |= (np.abs(w.CALI - w.BS) > 2.0).fillna(False).values
    if "DRHO" in w:
        bad |= (np.abs(w.DRHO) > 0.15).fillna(False).values
    return bad


# --------------------------------------------------------------------------- #
# 2. Petrofísica determinista
# --------------------------------------------------------------------------- #
def petrophysics(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    gr_cl, gr_sh = np.nanpercentile(w.GR, [5, 95])
    igr = np.clip((w.GR - gr_cl) / (gr_sh - gr_cl), 0, 1)
    w["VSH"] = np.clip(0.33 * (2 ** (2 * igr) - 1), 0, 1)          # Larionov
    phid = (RHO_MA - w.RHOB) / (RHO_MA - RHO_FL)
    sh = igr > 0.8
    phid_sh = np.nanmedian(phid[sh]) if sh.sum() >= 20 else np.nanmedian(phid)
    phid_sh = float(np.clip(phid_sh, 0, 0.3))
    w["PHIE"] = np.clip(phid - w.VSH * phid_sh, 0, 0.40)
    w["VP"] = FT2M / (w.DTC * 1e-6)
    w["RHO"] = w.RHOB * 1000.0
    w["AI"] = w.VP * w.RHO
    if "DTS" in w and w.DTS.notna().all():
        w["VS"] = FT2M / (w.DTS * 1e-6)
        w["SI"] = w.VS * w.RHO
    return w


# --------------------------------------------------------------------------- #
# 3. Conversión a tiempo y remuestreo
# --------------------------------------------------------------------------- #
def to_time(w: pd.DataFrame, dt: float = DT) -> pd.DataFrame:
    z = np.abs(w.Z_LOC.values) if w.Z_LOC.notna().all() else w.DEPTH_MD.values
    dz = np.diff(z, prepend=z[0])
    dz[0] = np.median(dz[1:])
    dz = np.clip(dz, 0, None)
    twt = 2.0 * np.cumsum(dz / w.VP.values)
    k = np.floor(twt / dt).astype(int)
    lith = w.FORCE_2020_LITHOFACIES_LITHOLOGY
    cols = {"k": k, "lnAI": np.log(w.AI.values), "lnRHO": np.log(w.RHO.values),
            "PHIE": w.PHIE.values, "VSH": w.VSH.values,
            "BAD": w.BAD.values.astype(float), "TVD": z,
            "SAND": (lith == 30000).astype(float).values,
            "SHALE": (lith == 65000).astype(float).values}
    if "SI" in w:
        cols["lnSI"] = np.log(w.SI.values)
    g = pd.DataFrame(cols).groupby("k").mean()
    full = pd.RangeIndex(g.index.min(), g.index.max() + 1)
    g = g.reindex(full).interpolate(limit_area="inside")
    g["t"] = g.index.values * dt
    return g.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Modelado convolucional
# --------------------------------------------------------------------------- #
def ricker(f0: float, dt: float = DT, length: float = 0.128,
           phase_deg: float = 0.0) -> np.ndarray:
    t = np.arange(-length / 2, length / 2 + dt / 2, dt)
    a = (np.pi * f0 * t) ** 2
    w = (1 - 2 * a) * np.exp(-a)
    if phase_deg:
        th = np.deg2rad(phase_deg)
        w = w * np.cos(th) - np.imag(hilbert(w)) * np.sin(th)
    return w


def lowpass(x: np.ndarray, fc: float, dt: float = DT, order: int = 4):
    b, a = butter(order, fc / (0.5 / dt))
    return filtfilt(b, a, x, padlen=min(len(x) - 1, 120))


def fatti_coefs(theta_deg: float, K) -> tuple:
    """Coeficientes de R(theta) = a*dlnAI + b*dlnSI + c*dlnRHO
    (Aki-Richards reescrita por Fatti et al.), con K = (Vs/Vp)^2."""
    th = np.deg2rad(theta_deg)
    t2, s2 = np.tan(th) ** 2, np.sin(th) ** 2
    return 0.5 * (1 + t2), -4.0 * K * s2, -(0.5 * t2 - 2.0 * K * s2)


def band_noise(n: int, wav: np.ndarray, rms: float, rng) -> np.ndarray:
    e = fftconvolve(rng.standard_normal(n), wav, mode="same")
    return e * rms / np.sqrt(np.mean(e ** 2))


def synth_acoustic(lnai, wav, snr, rng):
    ai = np.exp(lnai)
    r = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])                  # exacta
    s = fftconvolve(r, wav, mode="same")
    if np.isinf(snr):
        return s, s
    return s + band_noise(len(s), wav, np.sqrt(np.mean(s ** 2)) / snr, rng), s


def synth_elastic(lnai, lnsi, lnrho, wav, snr, rng, angles=ANGLES):
    """Trazas por ángulo con Aki-Richards/Fatti y K local = (Vs/Vp)^2 promediado
    en cada interfaz (el inversor usa K constante: discrepancia deliberada)."""
    vsvp2 = np.exp(2 * (lnsi - lnai))
    Kloc = 0.5 * (vsvp2[1:] + vsvp2[:-1])
    dai, dsi, drho = np.diff(lnai), np.diff(lnsi), np.diff(lnrho)
    ds, ss = [], []
    for th in angles:
        a, b, c = fatti_coefs(th, Kloc)
        s = fftconvolve(a * dai + b * dsi + c * drho, wav, mode="same")
        n = 0.0 if np.isinf(snr) else band_noise(len(s), wav,
                                                 np.sqrt(np.mean(s ** 2)) / snr, rng)
        ds.append(s + n)
        ss.append(s)
    return np.array(ds), np.array(ss)


# --------------------------------------------------------------------------- #
# 5. Inversión bayesiana linealizada
# --------------------------------------------------------------------------- #
def conv_matrix(wav: np.ndarray, n: int) -> np.ndarray:
    h = len(wav) // 2
    i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    k = i - j + h                                   # y[i] = sum_j r[j] w[i-j+h]
    W = np.zeros((n, n))
    m = (k >= 0) & (k < len(wav))
    W[m] = wav[k[m]]
    return W


def diff_matrix(n: int) -> np.ndarray:
    D = np.zeros((n - 1, n))
    D[np.arange(n - 1), np.arange(n - 1)] = -1
    D[np.arange(n - 1), np.arange(1, n)] = 1
    return D


def exp_corr(n: int, corr_len: float) -> np.ndarray:
    idx = np.arange(n)
    return np.exp(-np.abs(idx[:, None] - idx[None, :]) / corr_len)


def noise_cov(wav: np.ndarray, m: int, sig2: float) -> np.ndarray:
    """Covarianza del ruido limitado en banda: autocorrelación normalizada de la
    ondícula (Toeplitz) más una pepita blanca del 1 %."""
    ac = np.correlate(wav, wav, mode="full")
    ac = ac / ac.max()
    h = len(wav) - 1
    lag = np.abs(np.arange(m)[:, None] - np.arange(m)[None, :])
    Cn = np.where(lag <= h, ac[np.clip(h + lag, 0, 2 * h)], 0.0) * sig2
    return Cn + 0.01 * sig2 * np.eye(m)


def gaussian_update(G, d, m0, Cm, Cn):
    """x_post = m0 + Cm G^T (G Cm G^T + Cn)^-1 (d - G m0);  var = diag(C_post)."""
    CmGt = Cm @ G.T
    S = G @ CmGt + Cn
    K = solve(S, CmGt.T, assume_a="pos").T
    x = m0 + K @ (d - G @ m0)
    var = np.clip(np.diag(Cm) - np.einsum("ij,ji->i", K, CmGt.T), 0, None)
    return x, var


def invert_acoustic(d, x_lf, wav, sig_n2, sig_m2, corr_len):
    n = len(x_lf)
    G = 0.5 * conv_matrix(wav, n - 1) @ diff_matrix(n)
    Cm = sig_m2 * exp_corr(n, corr_len)
    return gaussian_update(G, d, x_lf, Cm, noise_cov(wav, n - 1, sig_n2))


def invert_elastic(D_ang, m_lf, wav, sig_n2, Sigma0, corr_len, K, angles=ANGLES):
    """Inversión AVO bayesiana linealizada para x = [lnAI, lnSI, lnRHO];
    covarianza a priori Sigma0 (3x3) ⊗ correlación exponencial; K constante."""
    n = m_lf.shape[1]
    WD = conv_matrix(wav, n - 1) @ diff_matrix(n)
    G = np.vstack([np.hstack([c * WD for c in fatti_coefs(th, K)]) for th in angles])
    Cm = np.kron(Sigma0, exp_corr(n, corr_len))
    m = n - 1
    Cn = np.zeros((len(angles) * m, len(angles) * m))
    for i in range(len(angles)):
        Cn[i * m:(i + 1) * m, i * m:(i + 1) * m] = noise_cov(wav, m, sig_n2[i])
    x, var = gaussian_update(G, D_ang.ravel(), m_lf.ravel(), Cm, Cn)
    return x.reshape(3, n), var.reshape(3, n)


def corr_length(resid_list, max_lag: int = 60) -> float:
    """Longitud de correlación (muestras) donde la autocorrelación media cae a 1/e."""
    acs = []
    for r in resid_list:
        r = r - r.mean()
        if len(r) <= max_lag + 10:
            continue
        acs.append([1.0] + [np.corrcoef(r[:-k], r[k:])[0, 1] for k in range(1, max_lag)])
    ac = np.nanmean(np.array(acs), axis=0)
    below = np.where(ac < np.exp(-1))[0]
    return float(below[0]) if len(below) else float(max_lag)


def forward_acoustic_lin(x, wav):
    return fftconvolve(0.5 * np.diff(x), wav, mode="same")


# --------------------------------------------------------------------------- #
# 6. Atributos
# --------------------------------------------------------------------------- #
def roll(x, win, fn):
    return getattr(pd.Series(x).rolling(win, center=True, min_periods=1), fn)().values


def seismic_attrs(d: np.ndarray, dt: float = DT) -> dict:
    dpad = np.r_[d, d[-1]]
    an = hilbert(dpad)
    fi = np.gradient(np.unwrap(np.angle(an)), dt) / (2 * np.pi)
    return {"amp": dpad, "envelope": np.abs(an), "cos_phase": np.cos(np.angle(an)),
            "inst_freq": roll(np.clip(fi, -100, 200), 11, "median")}


FEATURES = {
    "acoustic": ["lnAI_inv", "lnAI_rel", "amp", "envelope", "cos_phase",
                 "inst_freq", "lnAI_std21", "tvd"],
    "elastic": ["lnAI_inv", "lnAI_rel", "vpvs_inv", "lnSI_inv", "lambdarho",
                "murho", "envelope", "inst_freq", "lnAI_std21", "tvd"],
}
# Variables de la transformación lineal determinista (línea base)
LINEAR = {"acoustic": ["lnAI_inv"], "elastic": ["lnAI_inv", "vpvs_inv"]}
LINEAR_REF = {"acoustic": ["lnAI_true"], "elastic": ["lnAI_true", "vpvs_true"]}
# Restricciones de monotonía con motivación física (+1 creciente, -1 decreciente)
MONO = {"PHIE": {"lnAI_inv": -1}, "VSH": {"vpvs_inv": +1}}
