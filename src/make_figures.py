"""Figuras del artículo (300 dpi, texto en español, fuente Montserrat)."""
import json
import pickle

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

import cimis_core as C

RES, FIG = "/home/claude/cimis/results/", "/home/claude/cimis/figures/"
for f in ["Regular", "Bold", "SemiBold", "Italic"]:
    font_manager.fontManager.addfont(f"/home/claude/mont/fonts/ttf/Montserrat-{f}.ttf")
plt.rcParams.update({
    "font.family": ["Montserrat", "DejaVu Sans"], "font.size": 9, "axes.titlesize": 9,
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
    "grid.color": "#e4e3df", "grid.linewidth": 0.6, "savefig.dpi": 300,
    "axes.formatter.use_locale": False, "axes.unicode_minus": False,
})
BLUE, ORANGE, AQUA, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"


def comma(ax, axis="both", fmt="{:.2f}"):
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, p: fmt.format(v).replace(".", ","))
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(f)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(f)


# --------------------------------------------------------------------------- #
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.set_xlim(0, 100), ax.set_ylim(0, 66), ax.axis("off")
    boxes = [
        (2, 50, "M1. Ingesta y control\nde calidad", "LAS/CSV · rangos físicos\nhoyo (CALI–BS, DRHO)", True),
        (27, 50, "M2. Petrofísica\ndeterminista", "Vsh (Larionov)\nφe densidad-arcilla", True),
        (52, 50, "M3. Conversión\na tiempo", "TWT desde sónico\nremuestreo a 1 ms", True),
        (77, 50, "M4. Modelado\nsísmico", "convolucional\nAki-Richards/Fatti", True),
        (77, 27, "M5. Inversión\nbayesiana", "linealizada, acústica\ny AVO de 5° a 35°", True),
        (52, 27, "M6. Atributos\nelásticos y sísmicos", "ln AI, Vp/Vs, λρ, μρ\nenvolvente, frec. inst.", True),
        (27, 27, "M7. Aprendizaje\ncuantílico", "gradient boosting\nP10–P50–P90 + CQR", True),
        (2, 27, "M8. Validación\ny aceptación", "CV por pozo · ciegos\npruebas pareadas", True),
        (2, 4, "Sísmica de campo\nSEG-Y", "no evaluada en\neste estudio", False),
        (27, 4, "Muestreo MCMC y\nmodelos 3D", "propuesto; fuera del\nalcance experimental", False),
    ]
    W, H = 21, 15
    for x, y, t, s, impl in boxes:
        p = FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.4,rounding_size=1.2",
                           fc="#eef4fc" if impl else "white",
                           ec=BLUE if impl else GRAY, lw=1.0,
                           ls="-" if impl else (0, (3, 2)))
        ax.add_patch(p)
        ax.text(x + W / 2, y + H - 4.2, t, ha="center", va="center", fontsize=7.0,
                weight="semibold", color=INK)
        ax.text(x + W / 2, y + 3.6, s, ha="center", va="center", fontsize=6.2, color=MUTED)
    arr = dict(arrowstyle="-|>", color=MUTED, lw=0.9, mutation_scale=8)
    for (x0, y0), (x1, y1) in [((23.8, 57.5), (26.6, 57.5)), ((48.8, 57.5), (51.6, 57.5)),
                               ((73.8, 57.5), (76.6, 57.5)), ((87.5, 49.6), (87.5, 42.4)),
                               ((76.6, 34.5), (73.8, 34.5)), ((51.6, 34.5), (48.8, 34.5)),
                               ((26.6, 34.5), (23.8, 34.5))]:
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=arr)
    # retroalimentación
    ax.annotate("", (12.5, 49.6), (12.5, 42.6),
                arrowprops=dict(arrowstyle="-|>", color=ORANGE, lw=0.9, ls=(0, (3, 2)),
                                mutation_scale=8))
    ax.text(14, 46, "retroalimentación si\nno se cumple un criterio", fontsize=6, color=ORANGE,
            va="center")
    ax.plot([], [], color=BLUE, lw=6, alpha=.25, label="Módulo implementado y evaluado")
    ax.plot([], [], color=GRAY, lw=1, ls=(0, (3, 2)), label="Componente propuesto, no evaluado")
    ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.02))
    fig.savefig(FIG + "fig1_arquitectura.png", bbox_inches="tight")
    plt.close(fig)


def fig2_map():
    wa = pickle.load(open(RES + "wells_acoustic.pkl", "rb"))
    we = pickle.load(open(RES + "wells_elastic.pkl", "rb"))
    for w, tag in ((wa, "ac_base"), (we, "el_base")):
        sm = json.load(open(RES + f"summary_{tag}.json"))
        for split in ("train", "blind"):
            keep = set(sm[f"{split}_wells"])
            w[split] = {k: v for k, v in w[split].items() if k in keep}
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for split, mk, lab in (("train", "o", "entrenamiento"), ("blind", "^", "ciego")):
        xy = np.array([[w.X_LOC.mean(), w.Y_LOC.mean()] for w in wa[split].values()]) / 1000
        ax.scatter(xy[:, 0], xy[:, 1], s=26, marker=mk, facecolor="white", edgecolor=GRAY,
                   lw=1, label=f"Exp. 1 (acústico), {lab}: {len(xy)}", zorder=2)
        xy = np.array([[w.X_LOC.mean(), w.Y_LOC.mean()] for w in we[split].values()]) / 1000
        ax.scatter(xy[:, 0], xy[:, 1], s=26, marker=mk, color=BLUE if split == "train" else ORANGE,
                   edgecolor="white", lw=0.6, label=f"Exp. 2 (elástico), {lab}: {len(xy)}", zorder=3)
    ax.set_xlabel("Este UTM (km)"), ax.set_ylabel("Norte UTM (km)")
    ax.set_aspect("equal")
    ax.legend(loc="center left", frameon=False, fontsize=7, bbox_to_anchor=(1.02, 0.5))
    fig.savefig(FIG + "fig2_mapa.png", bbox_inches="tight")
    plt.close(fig)


def fig3_well(well="35/9-7"):
    D = pd.read_parquet(RES + "data_el_base.parquet")
    g = D[D.WELL == well].sort_values("t")
    # gathers sintéticos recalculados con la misma semilla no se almacenan: se
    # regeneran de forma determinista para ilustración
    we = pickle.load(open(RES + "wells_elastic.pkl", "rb"))
    T = C.to_time(we["blind"][well])
    rng = np.random.default_rng(7)
    Dg, _ = C.synth_elastic(T.lnAI.values, T.lnSI.values, T.lnRHO.values, C.ricker(30), 4.0, rng)
    t_ms = g.t.values * 1000
    fig, axs = plt.subplots(1, 5, figsize=(7.0, 5.0), sharey=True,
                            gridspec_kw={"width_ratios": [1, 1, 1.2, 1.2, 1.4]})
    axs[0].plot(g.VSH, t_ms, color=INK, lw=0.9)
    axs[0].set_xlabel("Vsh (fracción)"), axs[0].set_xlim(0, 1)
    axs[1].plot(g.PHIE, t_ms, color=INK, lw=0.9)
    axs[1].set_xlabel("φe (fracción)"), axs[1].set_xlim(0, 0.35)
    axs[2].plot(np.exp(g.lnAI_true) / 1e6, t_ms, color=INK, lw=0.9, label="registro")
    axs[2].plot(np.exp(g.lnAI_inv) / 1e6, t_ms, color=BLUE, lw=1.1, label="invertida")
    axs[2].plot(np.exp(g.lnAI_lf) / 1e6, t_ms, color=GRAY, lw=1.0, ls="--", label="baja frec.")
    axs[2].set_xlabel("AI (10⁶ kg m⁻² s⁻¹)")
    h, l = axs[2].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04))
    axs[3].plot(g.vpvs_true, t_ms, color=INK, lw=0.9, label="registro")
    axs[3].plot(g.vpvs_inv, t_ms, color=BLUE, lw=1.1, label="invertida")
    axs[3].set_xlabel("Vp/Vs")
    tt = np.arange(Dg.shape[1]) + 0.5 + g.t.values.min() * 1000
    sc = np.abs(Dg).max() * 0.6
    for i, th in enumerate(C.ANGLES):
        x = i * 1.0 + Dg[i] / sc
        axs[4].plot(x, tt, color=INK, lw=0.5)
        axs[4].fill_betweenx(tt, i, x, where=x > i, color=INK, lw=0, interpolate=True)
    axs[4].set_xticks(range(4), [f"{int(a)}°" for a in C.ANGLES])
    axs[4].set_xlabel("Ángulo de incidencia"), axs[4].grid(False)
    axs[0].set_ylabel("Tiempo doble relativo (ms)")
    axs[0].invert_yaxis()
    for a, lab in zip(axs, "abcde"):
        a.set_title(f"({lab})", loc="left", fontsize=8, color=MUTED)
    for a in axs[:2]:
        comma(a, "x", "{:.1f}")
    comma(axs[3], "x", "{:.1f}")
    comma(axs[2], "x", "{:.0f}")
    fig.tight_layout(w_pad=0.6)
    fig.savefig(FIG + "fig3_pozo_ciego.png", bbox_inches="tight")
    plt.close(fig)


def fig4_predictions(well="35/9-7"):
    P = pd.read_parquet(RES + "preds_el_base.parquet")
    fig, axs = plt.subplots(1, 2, figsize=(6.0, 5.0), sharey=True)
    for ax, tgt, lab, lim in zip(axs, ["PHIE", "VSH"], ["φe (fracción)", "Vsh (fracción)"],
                                 [(0, 0.35), (0, 1)]):
        g = P[(P.WELL == well) & (P.target == tgt) & (P.split == "blind")].sort_values("t")
        t = g.t.values * 1000
        ax.fill_betweenx(t, g.QGB_C_p10, g.QGB_C_p90, color=BLUE, alpha=0.18, lw=0,
                         label="CIMIS-ML P10–P90 (CQR)")
        ax.plot(g.y, t, color=INK, lw=0.9, label="registro (1 ms)")
        ax.plot(g.LIN_p50, t, color=ORANGE, lw=1.2, ls="--", label="transformación lineal")
        ax.plot(g.QGB_C_p50, t, color=BLUE, lw=1.2, label="CIMIS-ML P50")
        ax.set_xlim(*lim), ax.set_xlabel(lab)
        comma(ax, "x", "{:.2f}" if tgt == "PHIE" else "{:.1f}")
    axs[0].set_ylabel("Tiempo doble relativo (ms)"), axs[0].invert_yaxis()
    axs[0].set_title("(a)", loc="left", fontsize=8, color=MUTED)
    axs[1].set_title("(b)", loc="left", fontsize=8, color=MUTED)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    fig.savefig(FIG + "fig4_predicciones.png", bbox_inches="tight")
    plt.close(fig)


def fig5_summary():
    S = pd.read_csv(RES + "seed_summary_elastic.csv", header=[0, 1], index_col=[0, 1, 2])
    methods = ["LIN_A", "LIN", "QGB", "REF"]
    names = ["Lineal\nsolo AI", "Lineal\nAI + Vp/Vs", "CIMIS-ML\n(P50)", "Referencia\nregistro"]
    cols = [GRAY, ORANGE, BLUE, AQUA]
    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    for ax, (tgt, metric, ylab) in zip(axs, [("PHIE", "R2", "R² de φe"),
                                             ("VSH", "AUC", "AUC arena/lutita")]):
        for j, sp in enumerate(["cv", "blind"]):
            m = [S.loc[(tgt, sp, k), (metric, "mean")] for k in methods]
            e = [S.loc[(tgt, sp, k), (metric, "std")] for k in methods]
            x = np.arange(len(methods)) + (j - 0.5) * 0.36
            ax.bar(x, m, width=0.34, color=cols, alpha=1.0 if sp == "cv" else 0.45,
                   edgecolor="white", lw=1, yerr=e, capsize=2,
                   error_kw={"elinewidth": 0.7, "ecolor": MUTED})
        ax.axhline(0.5 if metric == "AUC" else 0, color=MUTED, lw=0.7)
        ax.set_xticks(range(len(methods)), names, fontsize=6.8)
        ax.set_ylabel(ylab)
        comma(ax, "y", "{:.2f}" if metric == "AUC" else "{:.1f}")
        if metric == "AUC":
            ax.set_ylim(0.4, 0.85)
    from matplotlib.patches import Patch
    fig.legend([Patch(color=MUTED), Patch(color=MUTED, alpha=0.45)],
               ["validación cruzada por pozo (22 pozos)", "pozos ciegos (7 pozos)"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.05))
    axs[0].set_title("(a)", loc="left", fontsize=8, color=MUTED)
    axs[1].set_title("(b)", loc="left", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(FIG + "fig5_resumen.png", bbox_inches="tight")
    plt.close(fig)


def fig6_sensitivity():
    tags = [("el_snr2", "S/R = 2"), ("el_base", "S/R = 4 (base)"), ("el_snr8", "S/R = 8"),
            ("el_snrinf", "sin ruido"), ("el_phase30", "fase +30°"),
            ("el_f0err", "f₀ −15 %"), ("el_flf2", "baja frec. 2 Hz"), ("el_flf8", "baja frec. 8 Hz")]
    rows = []
    for tag, lab in tags:
        s = json.load(open(RES + f"summary_{tag}.json"))
        cv = s["results"]["PHIE"]["cv"]
        rows.append((lab, cv["LIN"]["RMSE"], cv["QGB"]["RMSE"], cv["LIN"]["IS80"],
                     cv["QGB_C"]["IS80"], s["inv_true_corr_median"]))
    R = pd.DataFrame(rows, columns=["lab", "rl", "rq", "il", "iq", "inv"])
    y = np.arange(len(R))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 3.2), sharey=True,
                            gridspec_kw={"width_ratios": [1.2, 1.2, 0.8], "wspace": 0.12})
    for ax, (a, b, xl, fmt) in zip(axs[:2], [("rl", "rq", "RMSE de φe (fracción)", "{:.3f}"),
                                             ("il", "iq", "Interval score 80 % de φe", "{:.3f}")]):
        ax.hlines(y, R[a], R[b], color="#c3c2b7", lw=1.2)
        ax.scatter(R[a], y, color=ORANGE, s=28, marker="s", label="transformación lineal", zorder=3)
        ax.scatter(R[b], y, color=BLUE, s=30, marker="o", label="CIMIS-ML", zorder=3)
        ax.set_xlabel(xl)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        comma(ax, "x", fmt)
    axs[2].barh(y, R["inv"], color=GRAY, height=0.55)
    axs[2].set_xlabel("Correlación ln AI\ninvertida–registro")
    axs[2].set_xlim(0, 1)
    comma(axs[2], "x", "{:.1f}")
    axs[0].set_yticks(y, R["lab"])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.55, 1.07), ncol=2, frameon=False)
    for a, lab in zip(axs, "abc"):
        a.set_title(f"({lab})", loc="left", fontsize=8, color=MUTED)
    fig.savefig(FIG + "fig6_sensibilidad.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_architecture(); fig2_map(); fig3_well(); fig4_predictions(); fig5_summary(); fig6_sensitivity()
    print("ok")
