"""Figuras explicativas de las arquitecturas para la memoria.

Mantiene el estilo de la Ilustración 1 (esquema del autoencoder): Arial,
cajas redondeadas y la misma paleta gris / azul / naranja / verde.

Uso:  python generar_figuras.py   (deja los PNG y SVG en esta carpeta)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent
DPI = 300
ANCHO = 6.7  # pulgadas, igual que la Ilustración 1

# Paleta de la Ilustración 1
GRIS_F, GRIS_B = "#f3f2ee", "#9a9893"
AZUL_F, AZUL_B = "#eef4fc", "#2a78d6"
NAR_F, NAR_B = "#fde9e0", "#eb6834"
FLECHA = "#52514e"
TEXTO = "#0b0b0b"
TEXTO2 = "#55544f"

plt.rcParams.update({
    "font.family": "Arial",
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "svg.fonttype": "none",
})


# ---------------------------------------------------------------- utilidades
def caja(ax, cx, cy, w, h, texto="", fc=AZUL_F, ec=AZUL_B, fs=10, lw=1.6,
         ls="-", color=TEXTO, r=0.08, z=3, **kw):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
    if texto:
        ax.text(cx, cy, texto, ha="center", va="center", fontsize=fs,
                color=color, zorder=z + 1, **kw)


def op(ax, cx, cy, texto, r=0.24, fs=10):
    """Operación elemento a elemento (círculo)."""
    ax.add_patch(Circle((cx, cy), r, fc="white", ec=FLECHA, lw=1.3, zorder=3))
    ax.text(cx, cy - 0.01, texto, ha="center", va="center", fontsize=fs,
            color=TEXTO, zorder=4)


def traza(ax, pts, color=FLECHA, lw=1.4, flecha=True, z=1, ls="-"):
    """Polilínea; la flecha va en el último tramo."""
    xs, ys = zip(*pts)
    if flecha:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=z, ls=ls,
                solid_capstyle="butt")
        ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                    ls=ls, mutation_scale=10,
                                    shrinkA=0, shrinkB=0))
    else:
        ax.plot(xs, ys, color=color, lw=lw, zorder=z, ls=ls,
                solid_capstyle="butt")


def punto(ax, x, y, color=FLECHA):
    ax.add_patch(Circle((x, y), 0.07, fc=color, ec=color, zorder=2))


def puente(ax, x, y, fondo, horizontal=True, s=0.13):
    """Hueco blanco para marcar un cruce sin conexión."""
    if horizontal:
        ax.add_patch(Rectangle((x - s, y - 0.06), 2 * s, 0.12, fc=fondo,
                               ec="none", zorder=1.5))
    else:
        ax.add_patch(Rectangle((x - 0.06, y - s), 0.12, 2 * s, fc=fondo,
                               ec="none", zorder=1.5))


def guardar(fig, nombre):
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{nombre}.{ext}", dpi=DPI, bbox_inches="tight",
                    pad_inches=0.05, facecolor="white")
    plt.close(fig)


# ------------------------------------------------------- 1. LSTM frente a GRU
def celda_lstm(ax):
    YC, YB, YG, YM = 4.8, 1.2, 2.4, 3.6
    gw, gh = 0.9, 0.55
    caja(ax, 5.0, 3.0, 8.2, 4.6, fc=GRIS_F, ec=GRIS_B, r=0.25, z=0)

    # Estado de celda (cinta superior)
    traza(ax, [(0.15, YC), (2.16, YC)], color=NAR_B, lw=2.4)
    traza(ax, [(2.64, YC), (3.96, YC)], color=NAR_B, lw=2.4)
    traza(ax, [(4.44, YC), (9.85, YC)], color=NAR_B, lw=2.4)
    ax.text(0.15, YC + 0.3, r"$c_{t-1}$", fontsize=10, ha="left")
    ax.text(9.85, YC + 0.3, r"$c_t$", fontsize=10, ha="right")

    # Entradas: h_{t-1} y x_t se concatenan en un bus
    traza(ax, [(0.15, YB), (6.4, YB)], flecha=False)
    traza(ax, [(1.2, 0.0), (1.2, YB)], flecha=False)
    punto(ax, 1.2, YB)
    ax.text(0.15, YB - 0.35, r"$h_{t-1}$", fontsize=10, ha="left")
    ax.text(1.38, 0.1, r"$x_t$", fontsize=10, ha="left")

    puertas = [(2.4, r"$\sigma$", r"$f_t$"), (3.6, r"$\sigma$", r"$i_t$"),
               (4.8, "tanh", r"$\tilde{c}_t$"), (6.4, r"$\sigma$", r"$o_t$")]
    for x, t, lab in puertas:
        traza(ax, [(x, YB), (x, YG - gh / 2)])
        caja(ax, x, YG, gw, gh, t, fs=10 if t != "tanh" else 9)
        ax.text(x + 0.14, YG + gh / 2 + 0.28, lab, fontsize=9.5, ha="left",
                va="center", color=TEXTO2)
    for x in (2.4, 3.6, 4.8):
        punto(ax, x, YB)

    # Puerta de olvido
    traza(ax, [(2.4, YG + gh / 2), (2.4, YC - 0.24)])
    op(ax, 2.4, YC, "×")
    # Puerta de entrada por candidato
    traza(ax, [(3.6, YG + gh / 2), (3.6, YM), (3.96, YM)])
    traza(ax, [(4.8, YG + gh / 2), (4.8, YM), (4.44, YM)])
    op(ax, 4.2, YM, "×")
    traza(ax, [(4.2, YM + 0.24), (4.2, YC - 0.24)])
    op(ax, 4.2, YC, "+")
    # Salida
    punto(ax, 7.6, YC, color=NAR_B)
    traza(ax, [(7.6, YC), (7.6, 4.05)], color=NAR_B, lw=1.4)
    caja(ax, 7.6, 3.8, 0.9, 0.5, "tanh", fc="white", ec=FLECHA, lw=1.3, fs=9)
    traza(ax, [(7.6, 3.55), (7.6, YG + 0.24)])
    traza(ax, [(6.4 + gw / 2, YG), (7.36, YG)])
    op(ax, 7.6, YG, "×")
    traza(ax, [(7.6, YG - 0.24), (7.6, YB), (9.85, YB)])
    ax.text(9.85, YB - 0.35, r"$h_t$", fontsize=10, ha="right")


def celda_gru(ax):
    YH, YB, YG = 4.8, 1.2, 2.4
    gw, gh = 0.9, 0.55
    caja(ax, 5.0, 3.0, 8.2, 4.6, fc=GRIS_F, ec=GRIS_B, r=0.25, z=0)

    # Estado oculto (único estado de la GRU)
    traza(ax, [(0.15, YH), (6.16, YH)], color=AZUL_B, lw=2.4)
    traza(ax, [(6.64, YH), (7.96, YH)], color=AZUL_B, lw=2.4)
    traza(ax, [(8.44, YH), (9.85, YH)], color=AZUL_B, lw=2.4)
    ax.text(0.15, YH + 0.3, r"$h_{t-1}$", fontsize=10, ha="left")
    ax.text(9.85, YH + 0.3, r"$h_t$", fontsize=10, ha="right")

    # Bus h_{t-1} + x_t
    punto(ax, 1.3, YH, color=AZUL_B)
    traza(ax, [(1.3, YH), (1.3, YB), (8.2, YB)], flecha=False)
    traza(ax, [(1.7, 0.0), (1.7, YB)], flecha=False)
    punto(ax, 1.7, YB)
    ax.text(1.88, 0.1, r"$x_t$", fontsize=10, ha="left")

    for x, t, lab, d in [(2.4, r"$\sigma$", r"$z_t$", 1), (3.6, r"$\sigma$", r"$r_t$", -1)]:
        traza(ax, [(x, YB), (x, YG - gh / 2)])
        punto(ax, x, YB)
        caja(ax, x, YG, gw, gh, t)
        ax.text(x + d * 0.14, YG + gh / 2 + 0.28, lab, fontsize=9.5,
                ha="left" if d > 0 else "right", va="center", color=TEXTO2)

    # Puerta de reinicio: r_t ⊙ h_{t-1}
    traza(ax, [(3.6, YG + gh / 2), (3.6, 3.0), (4.66, 3.0)])
    punto(ax, 4.9, YH, color=AZUL_B)
    traza(ax, [(4.9, YH), (4.9, 3.24)])
    puente(ax, 4.9, 3.6, GRIS_F, horizontal=False)
    op(ax, 4.9, 3.0, "×")
    # Candidato h~_t, a partir de r_t ⊙ h_{t-1} y x_t
    traza(ax, [(5.14, 3.0), (7.4, 3.0), (7.4, YG + 0.1), (7.75, YG + 0.1)])
    traza(ax, [(8.2, YB), (8.2, YG - gh / 2)])
    caja(ax, 8.2, YG, gw, gh, "tanh", fs=9)
    ax.text(8.34, 3.05, r"$\tilde{h}_t$", fontsize=9.5,
            ha="left", va="center", color=TEXTO2)

    # Puerta de actualización: mezcla de h_{t-1} y h~_t
    traza(ax, [(2.4, YG + gh / 2), (2.4, 3.6), (7.96, 3.6)])
    punto(ax, 6.4, 3.6)
    traza(ax, [(6.4, 3.6), (6.4, 3.96)])
    op(ax, 6.4, 4.2, "1−", fs=8.5)
    traza(ax, [(6.4, 4.44), (6.4, YH - 0.24)])
    op(ax, 6.4, YH, "×")
    traza(ax, [(8.2, YG + gh / 2), (8.2, 3.36)])
    op(ax, 8.2, 3.6, "×")
    traza(ax, [(8.2, 3.84), (8.2, YH - 0.24)])
    op(ax, 8.2, YH, "+")


def fig_recurrentes():
    fig = plt.figure(figsize=(ANCHO, 2.5))
    axs = [fig.add_axes([0.0, 0.1, 0.49, 0.84]),
           fig.add_axes([0.51, 0.1, 0.49, 0.84])]
    celda_lstm(axs[0])
    celda_gru(axs[1])
    for ax, t in zip(axs, ["(a) Celda LSTM", "(b) Celda GRU"]):
        ax.set_xlim(-0.1, 10.1)
        ax.set_ylim(-0.25, 5.6)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(t, fontsize=11, pad=2)

    # Leyenda común, en unas coordenadas propias con aspecto 1:1
    lg = fig.add_axes([0.0, 0.0, 1.0, 0.1])
    lg.set_xlim(0, 20.4)
    lg.set_ylim(-0.38, 0.38)
    lg.set_aspect("equal")
    lg.axis("off")
    x = 1.0
    caja(lg, x, 0, 0.5, 0.36, fc=AZUL_F, ec=AZUL_B, lw=1.3, r=0.05)
    lg.text(x + 0.45, 0, "Capa con pesos", fontsize=9, va="center", color=TEXTO2)
    x = 5.0
    lg.add_patch(Circle((x, 0), 0.19, fc="white", ec=FLECHA, lw=1.2))
    lg.text(x + 0.45, 0, "Operación elemento a elemento", fontsize=9,
            va="center", color=TEXTO2)
    for x, c, t in [(12.3, NAR_B, "Estado de celda"),
                    (16.9, AZUL_B, "Estado oculto")]:
        lg.plot([x - 0.35, x + 0.25], [0, 0], color=c, lw=2.4)
        lg.text(x + 0.45, 0, t, fontsize=9, va="center", color=TEXTO2)
    guardar(fig, "fig_lstm_gru")


# ------------------------------------------------- 2. Conv1D frente a TCN
def red_conv(ax, n, capas, dil, causal, objetivo, titulo, etiquetas):
    """Dibuja n posiciones por capa y resalta el campo receptivo de `objetivo`."""
    ys = [0.0, 1.35, 2.7]
    xs = [i * 1.0 for i in range(n)]
    r = 0.2

    # Conexiones de cada salida a sus entradas (núcleo 3)
    def entradas(i, d):
        if causal:
            return [i - 2 * d, i - d, i]
        return [i - d, i, i + d]

    # Campo receptivo de la salida objetivo
    activos = [{objetivo}]
    for capa in range(capas, 0, -1):
        prev = set()
        for i in activos[0]:
            prev |= {j for j in entradas(i, dil[capa - 1]) if 0 <= j < n}
        activos.insert(0, prev)

    for capa in range(1, capas + 1):
        for i in range(n):
            for j in entradas(i, dil[capa - 1]):
                if not 0 <= j < n:
                    continue
                on = i in activos[capa] and j in activos[capa - 1]
                ax.plot([xs[j], xs[i]], [ys[capa - 1] + r, ys[capa] - r],
                        color=AZUL_B if on else "#d9d8d3",
                        lw=1.5 if on else 0.8, zorder=2 if on else 1)

    for capa in range(capas + 1):
        for i in range(n):
            on = i in activos[capa]
            if capa == capas and i == objetivo:
                fc, ec = NAR_F, NAR_B
            elif on:
                fc, ec = AZUL_F, AZUL_B
            else:
                fc, ec = "white", GRIS_B
            ax.add_patch(Circle((xs[i], ys[capa]), r, fc=fc, ec=ec,
                                lw=1.4 if on else 1.0, zorder=3))
        ax.text(-0.55, ys[capa], etiquetas[capa], ha="right", va="center",
                fontsize=9, color=TEXTO2)

    ax.annotate("", xy=(n - 0.4, -0.55), xytext=(-0.4, -0.55),
                arrowprops=dict(arrowstyle="-|>", color=FLECHA, lw=1.1,
                                mutation_scale=9, shrinkA=0, shrinkB=0))
    ax.text(n / 2 - 0.5, -0.85, "Posición en la ventana (tiempo)",
            ha="center", va="top", fontsize=8.5, color=TEXTO2)
    ax.set_title(titulo, fontsize=11, pad=4)
    ax.set_xlim(-2.6, n - 0.2)
    ax.set_ylim(-1.35, 3.05)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_convolucionales():
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 2.25))
    fig.subplots_adjust(wspace=0.04, left=0, right=1, top=0.9, bottom=0)
    red_conv(axs[0], 8, 2, [1, 1], causal=False, objetivo=4,
             titulo="(a) Conv1D",
             etiquetas=["Entrada", "Capa 1", "Capa 2"])
    red_conv(axs[1], 8, 2, [1, 2], causal=True, objetivo=7,
             titulo="(b) TCN (convolución causal dilatada)",
             etiquetas=["Entrada", "Capa 1\nd = 1", "Capa 2\nd = 2"])
    guardar(fig, "fig_conv1d_tcn")


# ---------------------------------------- 3. Atención completa frente a LogSparse
def mascara(ax, n, permitido, titulo, fila):
    for i in range(n):          # i = consulta (fila), j = clave (columna)
        for j in range(n):
            on = permitido(i, j)
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1,
                                   fc=AZUL_B if on else GRIS_F,
                                   ec="white", lw=1.5, zorder=1))
    ax.add_patch(Rectangle((0, n - 1 - fila), n, 1, fc="none", ec=NAR_B,
                           lw=2.2, zorder=3))
    for k in range(n):
        ax.text(k + 0.5, n + 0.15, str(k + 1), ha="center", va="bottom",
                fontsize=8.5, color=TEXTO2)
        ax.text(-0.15, n - 0.5 - k, str(k + 1), ha="right", va="center",
                fontsize=8.5, color=TEXTO2)
    ax.text(n / 2, n + 0.85, "Posición atendida", ha="center", va="bottom",
            fontsize=9, color=TEXTO2)
    ax.text(-0.95, n / 2, "Posición que se reconstruye", ha="center",
            va="center", rotation=90, fontsize=9, color=TEXTO2)
    ax.set_title(titulo, fontsize=11, pad=4)
    ax.set_xlim(-1.4, n + 0.1)
    ax.set_ylim(-0.1, n + 1.6)
    ax.set_aspect("equal")
    ax.axis("off")


def fig_atencion():
    n, fila = 8, 4  # se resalta la posición 5
    fig, axs = plt.subplots(1, 2, figsize=(ANCHO, 3.3))
    fig.subplots_adjust(wspace=0.12, left=0.02, right=0.98, top=0.93,
                        bottom=0.1)
    mascara(axs[0], n, lambda i, j: True, "(a) Transformer", fila)
    mascara(axs[1], n, lambda i, j: abs(i - j) in (0, 1, 2, 4, 8),
            "(b) LogSparse Transformer", fila)

    ly = 0.035
    for x, f, e, t in [(0.2, AZUL_B, AZUL_B, "Conexión permitida"),
                       (0.45, GRIS_F, GRIS_B, "Conexión eliminada"),
                       (0.7, "white", NAR_B, "Ejemplo, posición 5")]:
        fig.patches.append(Rectangle((x - 0.012, ly - 0.022), 0.024, 0.044,
                                     fc=f, ec=e, lw=1.4 if f == "white" else 0.6,
                                     transform=fig.transFigure, figure=fig))
        fig.text(x + 0.022, ly, t, fontsize=9, va="center", color=TEXTO2)
    guardar(fig, "fig_atencion_logsparse")


if __name__ == "__main__":
    fig_recurrentes()
    fig_convolucionales()
    fig_atencion()
    print("Figuras guardadas en", OUT)
