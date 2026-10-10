import numpy as np
import pytest

from tremor import estatistica as est


def test_separacao_perfeita():
    ref, alvo = np.arange(10), np.arange(10) + 100
    pv, r = est.mann_whitney(ref, alvo)
    assert r == pytest.approx(1.0)
    assert est.auc(ref, alvo) == pytest.approx(1.0)
    assert pv < 0.001


def test_ic_hanley_dentro_de_0_1():
    lo, hi = est.auc_ic_hanley(0.98, 5, 5)
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_reprodutivel():
    ref = np.random.default_rng(1).normal(0, 1, 30)
    alvo = np.random.default_rng(2).normal(1, 1, 30)
    a = est.auc_ic_bootstrap(ref, alvo, np.random.default_rng(42), B=200)
    b = est.auc_ic_bootstrap(ref, alvo, np.random.default_rng(42), B=200)
    assert a == b


def test_wilcoxon_pareado_sinal():
    x = np.arange(1, 21, dtype=float)
    pv, r = est.wilcoxon_pareado(x + 1, x)
    assert r == pytest.approx(1.0)
