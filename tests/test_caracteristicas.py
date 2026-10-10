import numpy as np
import pytest

from tremor.caracteristicas import calcular, fmed, kurtosis, mavsd

CFG = {"fs": 100.0, "tukey_alpha": 0.2}


def test_mavsd_reta_e_zero():
    # a segunda diferença de uma reta é nula
    assert mavsd(np.linspace(0, 5, 500)) == pytest.approx(0.0, abs=1e-12)


def test_mavsd_valor_conhecido():
    # x = [0, 1, 0, 1, ...] -> segunda diferença = ±2
    x = np.tile([0.0, 1.0], 50)
    assert mavsd(x) == pytest.approx(2.0)


def test_fmed_senoide():
    t = np.arange(2000) / 100
    assert fmed(np.sin(2 * np.pi * 5 * t), CFG) == pytest.approx(5.0, abs=0.1)


def test_kurtosis_normal_perto_de_3():
    x = np.random.default_rng(0).standard_normal(200_000)
    assert kurtosis(x) == pytest.approx(3.0, abs=0.05)


def test_calcular_rejeita_nome_desconhecido():
    with pytest.raises(KeyError):
        calcular(np.zeros(10), ["NaoExiste"], CFG)
