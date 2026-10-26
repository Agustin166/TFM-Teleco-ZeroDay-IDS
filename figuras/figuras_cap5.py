"""Figuras del capítulo 5, con el mismo estilo que generar_figuras.py.

Uso:  python figuras_cap5.py [carpeta_con_err_*.npz]
- fig_barrido_mirai: datos de 05_Barrido_Ventana (GRU, Mirai, con y sin src_ip).
- fig_histograma_error_denso: necesita err_nsl.npz, err_ds2os.npz y err_mirai.npz
  (error de reconstrucción del autoencoder denso, los genera la última celda
  de los cuadernos de 01_Autoencoder_Baseline).
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from generar_figuras import (ANCHO, AZUL_B, NAR_B, GRIS_B, FLECHA, TEXTO,
                             TEXTO2, guardar)

for lado in ("top", "right"):
    plt.rcParams[f"axes.spines.{lado}"] = False
plt.rcParams.update({"axes.edgecolor": GRIS_B, "axes.labelcolor": TEXTO2,
                     "xtick.color": TEXTO2, "ytick.color": TEXTO2,
                     "axes.formatter.use_locale": False})


def coma(x, d=1):
    return f"{x:.{d}f}".replace(".", ",")


def fig_barrido_mirai():
    W = [2, 3, 4, 5, 7, 10, 15, 20, 30]
    con = [0.8552, 0.8377, 0.8188, 0.8637, 0.8983, 0.9379, 0.9606, 0.9830, 1.0000]
    sin = [0.9211, 0.9026, 0.7723, 0.8211, 0.7319, 0.6247, 0.4122, 0.4151, 0.3666]
    fig, ax = plt.subplots(figsize=(ANCHO, 2.9))
    ax.axhline(0.5, color=GRIS_B, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(29.6, 0.515, "Azar (AUC = 0,5)", va="bottom", ha="right", fontsize=9, color=TEXTO2)
    ax.plot(W, con, color=NAR_B, lw=2, marker="o", ms=5, zorder=3,
            label="Con src_ip en la entrada (fuga)")
    ax.plot(W, sin, color=AZUL_B, lw=2, marker="o", ms=5, zorder=3,
            label="Sin src_ip (corregido)")
    ax.plot([3], [0.9026], marker="o", ms=11, mfc="none", mec=TEXTO, mew=1.4,
            zorder=4)
    ax.annotate("W = 3 seleccionado", xy=(3, 0.9026), xytext=(5.2, 1.0),
                fontsize=9, color=TEXTO, arrowprops=dict(arrowstyle="-",
                color=FLECHA, lw=0.9))
    ax.set_xticks(W)
    ax.set_xlim(1, 31.5)
    ax.set_ylim(0.3, 1.06)
    ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_yticklabels([coma(v) for v in ax.get_yticks()])
    ax.set_xlabel("Tamaño de ventana W", fontsize=10)
    ax.set_ylabel("AUC", fontsize=10)
    ax.tick_params(labelsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    guardar(fig, "fig_barrido_mirai_fuga")


def fig_histograma(carpeta):
    datos = [("NSL-KDD", "nsl"), ("DS2OS", "ds2os"), ("Mirai", "mirai")]
    fig, axs = plt.subplots(1, 3, figsize=(ANCHO, 2.45))
    fig.subplots_adjust(wspace=0.32, left=0.07, right=0.99, bottom=0.3, top=0.9)
    for ax, (titulo, clave) in zip(axs, datos):
        d = np.load(Path(carpeta) / f"err_{clave}.npz")
        e, y, thr = d["e_te"], d["y"], float(d["thr"])
        lo = max(e[e > 0].min(), 1e-6)
        bins = np.logspace(np.log10(lo), np.log10(e.max()), 45)
        for cls, col, lab in [(0, AZUL_B, "Normal"), (1, NAR_B, "Ataque")]:
            v = np.clip(e[y == cls], lo, None)
            w = np.ones_like(v) / len(v)
            ax.hist(v, bins=bins, weights=w, color=col, alpha=0.55,
                    edgecolor=col, linewidth=0.4, label=lab)
        ax.axvline(thr, color=TEXTO, lw=1.2, ls=(0, (4, 2.5)))
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        if np.log10(e.max() / lo) < 1.5:
            ticks = [v for v in (0.002, 0.005, 0.01, 0.02, 0.05) if lo <= v <= e.max()]
            ax.set_xticks(ticks)
            ax.set_xticklabels([coma(v, 3).rstrip("0") for v in ticks])
        ax.set_title(titulo, fontsize=10, color=TEXTO, pad=3)
        ax.tick_params(labelsize=8)
        ax.set_xlabel("Error de reconstrucción", fontsize=8.5)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _: coma(100 * v, 0) + " %"))
    axs[0].set_ylabel("Fracción de muestras", fontsize=8.5)
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color=TEXTO, lw=1.2, ls=(0, (4, 2.5))))
    l.append("Umbral (percentil 95)")
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=9)
    guardar(fig, "fig_histograma_error_denso")


def fig_formas_ebm(carpeta="figuras_ebm"):
    """Funciones de forma del EBM sustituto del autoencoder denso (NetFlow real).

    Los CSV se exportan en el servidor con explain_global(); cada fila es el
    límite inferior de un tramo y su contribución (log-odds) a la alerta.
    """
    import pandas as pd
    paneles = [("dst_port", "(a) dst_port", "Puerto de destino", True, None),
               ("wan_l4_port", "(b) wan_l4_port", "Puerto del extremo remoto", True, None),
               ("sum_pkts", "(c) sum_pkts", "Valor de sum_pkts", False, None),
               ("tcp_flags", "(d) tcp_flags", "Valor codificado de tcp_flags", False, None)]
    fig, axs = plt.subplots(2, 2, figsize=(ANCHO, 4.6))
    fig.subplots_adjust(wspace=0.25, hspace=0.62, left=0.09, right=0.99,
                        bottom=0.1, top=0.94)
    for ax, (col, titulo, xlab, puertos, _) in zip(axs.flat, paneles):
        d = pd.read_csv(Path(carpeta) / f"forma_{col}.csv")
        x = d["limite"].to_numpy(dtype=float)
        s = d["score"].to_numpy(dtype=float)
        fin = 65535.0 if puertos else x[-1] + (x[-1] - x[0]) * 0.03
        x = np.append(x, fin)
        s = np.append(s, s[-1])
        if puertos:
            x[0] = 1.0
            ax.set_xscale("log")
            ax.set_xlim(1, 65535)
            ax.set_xticks([1, 10, 100, 1000, 10000])
            ax.set_xticklabels(["1", "10", "100", "1.000", "10.000"])
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
        else:
            ax.set_xlim(x[0], x[-1])
        ax.axhline(0, color=GRIS_B, lw=0.9, zorder=1)
        ax.fill_between(x, np.clip(s, 0, None), 0, step="post", color=NAR_B,
                        alpha=0.35, lw=0, zorder=2)
        ax.fill_between(x, np.clip(s, None, 0), 0, step="post", color=AZUL_B,
                        alpha=0.35, lw=0, zorder=2)
        ax.step(x, s, where="post", color=TEXTO, lw=1, zorder=3)
        ax.set_title(titulo, fontsize=10, color=TEXTO, pad=3, loc="left")
        ax.set_xlabel(xlab, fontsize=8.5)
        ax.tick_params(labelsize=8)
        dec = 1 if np.abs(s).max() < 2 else 0
        ax.yaxis.set_major_formatter(plt.FuncFormatter(
            lambda v, _, dec=dec: coma(v, dec).replace("-0,0", "0,0")))
    for ax in axs[:, 0]:
        ax.set_ylabel("Contribución a la alerta", fontsize=8.5)

    a, b, c, dd = axs.flat
    a.annotate("Puerto 23", xy=(24, 2.26), xytext=(120, 1.9), fontsize=8.5,
               color=TEXTO, arrowprops=dict(arrowstyle="-", color=FLECHA, lw=0.9))
    dd.annotate("181", xy=(181, 0.91), xytext=(205, 0.75), fontsize=8.5,
                color=TEXTO, arrowprops=dict(arrowstyle="-", color=FLECHA, lw=0.9))
    guardar(fig, "fig_formas_ebm")


if __name__ == "__main__":
    fig_barrido_mirai()
    if len(sys.argv) > 1 and sys.argv[1] == "ebm":
        fig_formas_ebm()
    elif len(sys.argv) > 1:
        fig_histograma(sys.argv[1])
