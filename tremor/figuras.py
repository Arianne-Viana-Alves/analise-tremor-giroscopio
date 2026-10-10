"""Figuras: boxplots por característica e exemplos de sinais brutos."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .caracteristicas import ESCALA_LOG, UNIDADE

CORES = ("#0072B2", "#D55E00")  # par acessível (azul / laranja)
CONDICOES = (("sem", "Sem filtro"), ("com", "Com filtro"))


def _fmt_p(pv):
    return "p < 0,001" if pv < 0.001 else "p = " + format(pv, ".3f").replace(".", ",")


def boxplots(df, grupos, rotulos, caracteristicas, pvals, pasta, banda_txt=""):
    """Um PNG por característica, com painéis sem e com filtro. Caixa
    (mediana, Q1-Q3, bigodes até 1,5xIQR) e um ponto por indivíduo."""
    cores = dict(zip(grupos, CORES))
    for feat in caracteristicas:
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), dpi=200)
        rng = np.random.default_rng(1)
        for ax, (cond, titulo) in zip(axes, CONDICOES):
            for i, g in enumerate(grupos):
                v = df.loc[df.grupo == g, f"{feat}_{cond}"].values
                ax.boxplot(v, positions=[i], widths=0.5, showfliers=False,
                           patch_artist=True,
                           boxprops=dict(facecolor="none", edgecolor="black"),
                           medianprops=dict(color="black", lw=1.8),
                           whiskerprops=dict(color="black"),
                           capprops=dict(color="black"))
                ax.scatter(i + rng.uniform(-0.2, 0.2, len(v)), v, s=7,
                           color=cores[g], alpha=0.45, lw=0, zorder=3)
            if ESCALA_LOG.get(feat, False):
                ax.set_yscale("log")
            ax.set_xticks(range(len(grupos)))
            ax.set_xticklabels([f"{rotulos.get(g, g)}\n(n={int((df.grupo == g).sum())})"
                                for g in grupos])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            pv = pvals[(feat, cond)]
            extra = f" ({banda_txt})" if cond == "com" and banda_txt else ""
            ax.set_title(f"{titulo}{extra}  ·  {_fmt_p(pv)} "
                         f"{'*' if pv < 0.05 else '(ns)'}", fontsize=9)
        unidade = UNIDADE.get(feat)
        ylabel = feat + (" (escala log)" if ESCALA_LOG.get(feat) else
                         (f" ({unidade})" if unidade else ""))
        axes[0].set_ylabel(ylabel)
        fig.tight_layout()
        fig.savefig(os.path.join(pasta, f"boxplot_{feat.lower()}.png"))
        plt.close(fig)


def exemplos(df, sinais, grupos, rotulos, fs, pasta, ids=None, semente=42,
             rotulo_y="Magnitude do sinal"):
    """Grade 2x2: dois indivíduos por grupo. Se ``ids`` não for dado, sorteia
    dois por grupo com a semente indicada."""
    cores = dict(zip(grupos, CORES))
    if ids:
        escolha = {g: [i for i in ids if df.loc[i, "grupo"] == g][:2] for g in grupos}
    else:
        rng = np.random.default_rng(semente)
        escolha = {g: list(rng.choice(df.index[df.grupo == g], 2, replace=False))
                   for g in grupos}
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.2), dpi=200, sharex=True)
    for i, g in enumerate(grupos):
        for j, vid in enumerate(escolha[g]):
            x = sinais[vid]
            ax = axes[i, j]
            ax.plot(np.arange(len(x)) / fs, x, lw=0.6, color=cores[g])
            ax.set_title(f"{rotulos.get(g, g)} - Exemplo {j + 1} (id {vid})", fontsize=8)
            if j == 0:
                ax.set_ylabel(rotulo_y)
            if i == 1:
                ax.set_xlabel("Tempo (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(pasta, "sinais_exemplos.png"))
    plt.close(fig)
