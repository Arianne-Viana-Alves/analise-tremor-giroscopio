"""Teste ponta a ponta: amostra -> extrair -> analisar."""
import pandas as pd

from tremor.cli import main


def test_pipeline_completo(base_sintetica):
    d = base_sintetica
    main(["amostra", "--pacientes", str(d / "patients"), "--raw", str(d / "raw"),
          "--saida", str(d / "amostra.csv")])
    amostra = pd.read_csv(d / "amostra.csv")
    assert amostra.grupo.value_counts().to_dict() == {"Parkinson": 8, "Saudavel": 8}

    main(["extrair", "--raw", str(d / "raw"), "--amostra", str(d / "amostra.csv"),
          "--out", str(d / "saida")])
    res = pd.read_csv(d / "saida" / "resultados_por_individuo.csv")
    for col in ("MAVSD_sem", "MAVSD_com", "FMED_sem", "FMED_com", "Kurtosis_sem", "Kurtosis_com"):
        assert col in res.columns

    main(["analisar", "--res", str(d / "saida" / "resultados_por_individuo.csv"),
          "--out", str(d / "saida"), "--bootstrap", "100"])
    comp = pd.read_csv(d / "saida" / "comparacao_grupos.csv")
    mavsd = comp[(comp.caracteristica == "MAVSD") & (comp.filtro == "sem")].iloc[0]
    assert mavsd.auc > 0.9  # o grupo sintético "Parkinson" tem mais variabilidade
    for f in ("relatorio.txt", "boxplot_mavsd.png", "sinais_exemplos.png"):
        assert (d / "saida" / f).exists()


def test_selecao_balanceada_e_outro_grupo(base_sintetica):
    d = base_sintetica
    main(["amostra", "--pacientes", str(d / "patients"),
          "--grupos", "Healthy=Controle", "Other Movement Disorders=Outros",
          "--balancear", "--saida", str(d / "a.csv")])
    a = pd.read_csv(d / "a.csv")
    assert a.grupo.value_counts().to_dict() == {"Controle": 2, "Outros": 2}


def test_acelerometro_eixo_unico(base_sintetica):
    d = base_sintetica
    main(["amostra", "--pacientes", str(d / "patients"), "--saida", str(d / "a.csv")])
    main(["extrair", "--raw", str(d / "raw"), "--amostra", str(d / "a.csv"),
          "--out", str(d / "o"), "--sensor", "acelerometro", "--combinacao", "z",
          "--tipo-filtro", "lowpass", "--banda", "15", "--caracteristicas", "MAVSD",
          "--sem-figuras"])
    res = pd.read_csv(d / "o" / "resultados_por_individuo.csv")
    assert {"MAVSD_sem", "MAVSD_com"} <= set(res.columns)
    assert "FMED_sem" not in res.columns
