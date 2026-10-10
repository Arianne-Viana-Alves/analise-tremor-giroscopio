"""Leitura e pré-processamento de sinais inerciais.

Formato esperado de cada arquivo (padrão da base PADS):
    texto delimitado, uma linha por amostra, colunas
    0 = tempo | 1-3 = acelerômetro x/y/z | 4-6 = giroscópio x/y/z
Outras disposições de colunas são aceitas via o parâmetro ``colunas``.
"""
import numpy as np
from scipy import signal

# Colunas de cada sensor no layout PADS
SENSORES = {
    "acelerometro": [1, 2, 3],
    "giroscopio": [4, 5, 6],
}


def carregar(caminho, delimitador=","):
    """Lê o arquivo bruto como matriz (amostras x colunas)."""
    return np.loadtxt(caminho, delimiter=delimitador, ndmin=2)


def combinar_eixos(dados, colunas, combinacao="magnitude"):
    """Reduz os eixos escolhidos a um único sinal.

    combinacao:
        "magnitude"  -> sqrt(x² + y² + z²)  (usado no artigo)
        "x", "y", "z" -> um eixo isolado (posição na lista ``colunas``)
    """
    a = dados[:, list(colunas)]
    if combinacao == "magnitude":
        return np.sqrt((a ** 2).sum(axis=1))
    eixos = {"x": 0, "y": 1, "z": 2}
    if combinacao in eixos:
        return a[:, eixos[combinacao]]
    raise ValueError(f"combinacao desconhecida: {combinacao!r}")


def descartar_inicio(x, n):
    """Remove as n primeiras amostras (ex.: vibração de notificação do relógio)."""
    return x[n:]


def filtrar(x, fs, banda=(1.0, 10.0), ordem=4, tipo="bandpass"):
    """Butterworth de fase zero (filtfilt). ``banda`` é uma tupla (passa-faixa /
    rejeita-faixa) ou um número (passa-baixa / passa-alta). A ordem efetiva é o
    dobro de ``ordem`` por causa da filtragem ida e volta."""
    b, a = signal.butter(ordem, banda, btype=tipo, fs=fs)
    return signal.filtfilt(b, a, x)


def preparar(caminho, cfg):
    """Pipeline completo de um arquivo: leitura, combinação de eixos e descarte.
    Retorna o sinal sem filtro; o filtrado é obtido com ``filtrar``."""
    dados = carregar(caminho, cfg["delimitador"])
    colunas = cfg["colunas"] or SENSORES[cfg["sensor"]]
    x = combinar_eixos(dados, colunas, cfg["combinacao"])
    return descartar_inicio(x, cfg["descarte"])
