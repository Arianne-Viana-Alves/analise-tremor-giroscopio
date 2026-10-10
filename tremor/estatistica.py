"""Testes estatísticos para comparar dois grupos independentes e o efeito
pareado da filtragem.

Convenção: ``ref`` é o grupo de referência (ex.: controles) e ``alvo`` o grupo
de interesse (ex.: Parkinson). r > 0 e AUC > 0,5 indicam valores maiores no
grupo alvo.
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import RobustScaler


def mediana_iq(v):
    """(mediana, Q1, Q3)."""
    q1, q2, q3 = np.percentile(v, [25, 50, 75])
    return q2, q1, q3


def mann_whitney(ref, alvo):
    """Mann-Whitney bilateral. Retorna (p, r rank-biserial), com
    r = 1 - 2U/(n1 n2) e U calculado para o grupo de referência."""
    u, pv = stats.mannwhitneyu(ref, alvo, alternative="two-sided")
    return pv, 1 - 2 * u / (len(ref) * len(alvo))


def auc(ref, alvo):
    """Área sob a curva ROC usando o próprio valor da característica como
    escore (grupo alvo = classe positiva)."""
    y = np.r_[np.zeros(len(ref)), np.ones(len(alvo))]
    return roc_auc_score(y, np.r_[ref, alvo])


def auc_ic_hanley(a, n1, n2):
    """IC95% da AUC pela aproximação de Hanley-McNeil."""
    q1, q2 = a / (2 - a), 2 * a ** 2 / (1 + a)
    se = np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a ** 2)
                  + (n2 - 1) * (q2 - a ** 2)) / (n1 * n2))
    return max(0.0, a - 1.96 * se), min(1.0, a + 1.96 * se)


def auc_ic_bootstrap(ref, alvo, rng, B=5000):
    """IC95% percentil da AUC por bootstrap estratificado (reamostra cada
    grupo separadamente). ``rng`` é um numpy Generator compartilhado, para que
    a sequência de sorteios seja reprodutível."""
    ref, alvo = np.asarray(ref), np.asarray(alvo)
    y = np.r_[np.zeros(len(ref)), np.ones(len(alvo))]
    vals = np.empty(B)
    for b in range(B):
        i = rng.integers(0, len(ref), len(ref))
        j = rng.integers(0, len(alvo), len(alvo))
        vals[b] = roc_auc_score(y, np.r_[ref[i], alvo[j]])
    return tuple(np.percentile(vals, [2.5, 97.5]))


def shapiro(v):
    """p do Shapiro-Wilk (NaN se n < 3)."""
    v = np.asarray(v)
    return stats.shapiro(v).pvalue if len(v) >= 3 else float("nan")


def wilcoxon_pareado(x, y):
    """Wilcoxon de postos sinalizados e r rank-biserial pareado
    (soma dos postos positivos - negativos) / soma total."""
    pv = stats.wilcoxon(x, y).pvalue
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    rk = stats.rankdata(np.abs(d))
    r = (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
    return pv, r


def loocv_auc(X, y):
    """Regressão logística com validação leave-one-out; o RobustScaler é
    ajustado dentro de cada dobra (sem vazamento). Retorna a AUC das
    probabilidades fora da amostra."""
    X = np.asarray(X, dtype=float)
    prob = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        sc = RobustScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        prob[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, prob)
