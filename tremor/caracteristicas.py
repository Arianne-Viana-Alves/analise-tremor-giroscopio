"""Características extraídas de um sinal 1-D.

Toda função recebe (x, cfg) e devolve um float. Para acrescentar uma nova
característica, escreva a função e registre-a em ``CARACTERISTICAS``; ela passa
a poder ser pedida por nome em ``--caracteristicas``.
"""
import numpy as np
from scipy import signal, stats


def mavsd(x, cfg=None):
    """Média do valor absoluto da segunda diferença:
    mean(|x[i] - 2x[i-1] + x[i-2]|)."""
    x = np.asarray(x, dtype=float)
    return float(np.mean(np.abs(x[2:] - 2 * x[1:-1] + x[:-2])))


def fmed(x, cfg):
    """Frequência mediana: primeira frequência da grade da FFT em que a energia
    acumulada atinge 50% do total. Remoção de tendência linear, janela de Tukey
    (alpha = cfg['tukey_alpha']), FFT de segmento único (sem Welch),
    energia = |X|²."""
    x = signal.detrend(np.asarray(x, dtype=float), type="linear")
    x = x * signal.windows.tukey(len(x), cfg["tukey_alpha"])
    energia = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / cfg["fs"])
    acum = np.cumsum(energia)
    return float(f[np.searchsorted(acum, acum[-1] / 2)])


def kurtosis(x, cfg=None):
    """Curtose de Pearson (populacional, não excesso): normal = 3."""
    return float(stats.kurtosis(x, fisher=False))


CARACTERISTICAS = {
    "MAVSD": mavsd,
    "FMED": fmed,
    "Kurtosis": kurtosis,
}

# Escala logarítmica nos boxplots (características muito assimétricas)
ESCALA_LOG = {"MAVSD": True, "Kurtosis": True}
UNIDADE = {"FMED": "Hz"}


def calcular(x, nomes, cfg):
    """Dicionário {nome: valor} para as características pedidas."""
    desconhecidas = [n for n in nomes if n not in CARACTERISTICAS]
    if desconhecidas:
        raise KeyError(f"características não registradas: {desconhecidas}. "
                       f"Disponíveis: {list(CARACTERISTICAS)}")
    return {n: CARACTERISTICAS[n](x, cfg) for n in nomes}
