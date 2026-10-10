"""Interface de linha de comando.

    python -m tremor amostra   ...  monta a tabela de indivíduos a partir dos metadados
    python -m tremor extrair   ...  calcula as características de cada sinal
    python -m tremor analisar  ...  estatística e figuras sobre a tabela de características

Use ``python -m tremor <comando> -h`` para ver todas as opções. Qualquer opção
pode vir de um arquivo JSON via ``--config`` (as chaves são os nomes das
opções com "_" no lugar de "-"); opções digitadas na linha de comando têm
prioridade sobre o arquivo.
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

from . import estatistica as est
from . import figuras
from .caracteristicas import CARACTERISTICAS, calcular
from .sinais import SENSORES, filtrar, preparar

# Padrão usado no artigo para sinalizar diagnósticos atípicos no grupo Parkinson
REGEX_ATIPICOS_PADS = (r"essential|atypical|atypisch|msa|multiple system|vascular|"
                       r"psp|supranuclear|lewy|dd:|dd,")


# ------------------------------------------------------------------ utilidades
def _lista(txt, tipo=str):
    if txt is None or txt == "":
        return []
    if isinstance(txt, (list, tuple)):
        return [tipo(t) for t in txt]
    return [tipo(t.strip()) for t in str(txt).split(",") if t.strip()]


def _mapa(pares):
    """['A=B', 'C=D'] -> {'A': 'B', 'C': 'D'}"""
    out = {}
    for p in pares or []:
        k, _, v = p.partition("=")
        out[k] = v or k
    return out


def _id(v, largura):
    s = str(v).strip()
    return s.zfill(largura) if s.isdigit() else s


# ------------------------------------------------------------------- amostra
def cmd_amostra(a):
    arquivos = sorted(glob.glob(os.path.join(a.pacientes, a.padrao_paciente)))
    if not arquivos:
        sys.exit(f"Nenhum arquivo '{a.padrao_paciente}' em {a.pacientes}")
    mapa = _mapa(a.grupos)
    linhas = []
    for p in arquivos:
        m = json.load(open(p, encoding="utf-8"))
        cond = m.get(a.campo_condicao)
        if cond not in mapa:
            continue
        vid = _id(m.get("id", re.findall(r"\d+", os.path.basename(p))[-1]), a.largura_id)
        linhas.append({
            "id": vid, "grupo": mapa[cond], "idade": m.get("age"),
            "sexo": m.get("gender"),
            "arquivo": a.padrao_arquivo.format(id=vid, tarefa=a.tarefa, punho=a.punho),
            "comentario_clinico": (m.get("disease_comment") or "").replace("\n", " "),
        })
    df = pd.DataFrame(linhas)
    if df.empty:
        sys.exit("Nenhum indivíduo com as condições pedidas em --grupos.")

    if a.raw:
        existe = df.arquivo.map(lambda f: os.path.exists(os.path.join(a.raw, f)))
        if (~existe).any():
            print(f"Aviso: {int((~existe).sum())} indivíduos sem arquivo em {a.raw} foram removidos.")
        df = df[existe]

    if a.ids:
        ids = {_id(i, a.largura_id) for i in _lista(a.ids)}
        df = df[df.id.isin(ids)]
    if a.excluir_ids:
        df = df[~df.id.isin({_id(i, a.largura_id) for i in _lista(a.excluir_ids)})]

    n = a.n_por_grupo
    if a.balancear:
        menor = df.grupo.value_counts().min()
        n = min(n, menor) if n else menor
    if n:
        partes = []
        for g, sub in df.groupby("grupo", sort=False):
            sub = sub.sort_values("id")
            if a.selecao == "aleatoria":
                sub = sub.sample(n=min(n, len(sub)), random_state=a.semente).sort_values("id")
            else:
                sub = sub.head(n)
            partes.append(sub)
        df = pd.concat(partes)

    df = df.sort_values(["grupo", "id"])
    df.to_csv(a.saida, index=False)
    print(f"{len(df)} indivíduos salvos em {a.saida}")
    print(df.grupo.value_counts().to_string())


# ------------------------------------------------------------------- extrair
def _cfg_sinal(a):
    return {
        "fs": a.fs, "descarte": a.descarte, "sensor": a.sensor,
        "colunas": _lista(a.colunas, int) or None, "combinacao": a.combinacao,
        "delimitador": a.delimitador, "tukey_alpha": a.tukey_alpha,
    }


def cmd_extrair(a):
    cfg = _cfg_sinal(a)
    banda = tuple(_lista(a.banda, float))
    banda = banda[0] if len(banda) == 1 else banda
    nomes = _lista(a.caracteristicas)
    amostra = pd.read_csv(a.amostra, dtype={"id": str})
    amostra["id"] = amostra["id"].map(lambda v: _id(v, a.largura_id))
    os.makedirs(a.out, exist_ok=True)

    linhas, sinais = [], {}
    for _, r in amostra.iterrows():
        arq = r["arquivo"] if "arquivo" in r and isinstance(r["arquivo"], str) else \
            a.padrao_arquivo.format(id=r["id"], tarefa=a.tarefa, punho=a.punho)
        x = preparar(os.path.join(a.raw, arq), cfg)
        y = filtrar(x, a.fs, banda, a.ordem, a.tipo_filtro)
        sinais[r["id"]] = x
        lin = {k: r[k] for k in amostra.columns if k != "arquivo"}
        for cond, s in (("sem", x), ("com", y)):
            for k, v in calcular(s, nomes, cfg).items():
                lin[f"{k}_{cond}"] = v
        linhas.append(lin)

    df = pd.DataFrame(linhas).set_index("id")
    # colunas de características agrupadas por nome (MAVSD_sem, MAVSD_com, ...)
    feats = [f"{n}_{c}" for n in nomes for c in ("sem", "com")]
    df = df[[c for c in df.columns if c not in feats] + feats]
    saida = os.path.join(a.out, "resultados_por_individuo.csv")
    df.to_csv(saida)

    params = {**cfg, "banda": banda, "ordem": a.ordem, "tipo_filtro": a.tipo_filtro,
              "caracteristicas": nomes, "n_amostras_por_sinal": len(next(iter(sinais.values())))}
    json.dump(params, open(os.path.join(a.out, "parametros_extracao.json"), "w"),
              indent=2, ensure_ascii=False, default=str)

    if not a.sem_figuras and df.grupo.nunique() >= 2:
        grupos = list(dict.fromkeys(df.grupo))[:2]
        sensor = "giroscópio (rad/s)" if cfg["sensor"] == "giroscopio" else "acelerômetro"
        figuras.exemplos(df, sinais, grupos, _mapa(a.rotulos), a.fs, a.out,
                         ids=[_id(i, a.largura_id) for i in _lista(a.exemplos)] or None,
                         semente=a.semente, rotulo_y=f"Magnitude {sensor}")
    print(f"{len(df)} sinais processados -> {saida}")


# ------------------------------------------------------------------ analisar
def cmd_analisar(a):
    df = pd.read_csv(a.res, dtype={"id": str}).set_index("id")
    ref, alvo = a.ref, a.alvo
    df = df[df.grupo.isin([ref, alvo])]
    R, A = df[df.grupo == ref], df[df.grupo == alvo]
    if len(R) == 0 or len(A) == 0:
        sys.exit(f"Grupos '{ref}' e/ou '{alvo}' não encontrados. Presentes: {sorted(df.grupo.unique())}")
    feats = _lista(a.caracteristicas) or [c[:-4] for c in df.columns if c.endswith("_sem")]
    principal = a.principal if a.principal in feats else feats[0]
    rotulos = _mapa(a.rotulos)
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.semente)

    L = []
    w = L.append
    w(f"N total = {len(df)}  ({ref} = {len(R)}, {alvo} = {len(A)})")
    w(f"Características: {', '.join(feats)}  |  principal: {principal}")

    # 1. Descritiva, Mann-Whitney, r, AUC (IC Hanley e bootstrap)
    w(f"\n== Medianas [Q1-Q3], Mann-Whitney bilateral, r rank-biserial, AUC "
      f"(IC95% Hanley | bootstrap B={a.bootstrap}) ==")
    resumo, pvals = [], {}
    for f in feats:
        for c in ("sem", "com"):
            col = f"{f}_{c}"
            r_, a_ = R[col].values, A[col].values
            mr, ma = est.mediana_iq(r_), est.mediana_iq(a_)
            pv, rr = est.mann_whitney(r_, a_)
            au = est.auc(r_, a_)
            hl = est.auc_ic_hanley(au, len(r_), len(a_))
            bt = est.auc_ic_bootstrap(r_, a_, rng, a.bootstrap) if a.bootstrap else (np.nan, np.nan)
            pvals[(f, c)] = pv
            resumo.append(dict(caracteristica=f, filtro=c,
                               mediana_ref=mr[0], q1_ref=mr[1], q3_ref=mr[2],
                               mediana_alvo=ma[0], q1_alvo=ma[1], q3_alvo=ma[2],
                               p_mann_whitney=pv, r_rank_biserial=rr, auc=au,
                               auc_ic_hanley_inf=hl[0], auc_ic_hanley_sup=hl[1],
                               auc_ic_boot_inf=bt[0], auc_ic_boot_sup=bt[1]))
            w(f"{f:9s} {c}: {ref} {mr[0]:.6g} [{mr[1]:.6g}-{mr[2]:.6g}] | "
              f"{alvo} {ma[0]:.6g} [{ma[1]:.6g}-{ma[2]:.6g}] | p={pv:.4g} r={rr:.3f} "
              f"AUC={au:.3f} (Hanley {hl[0]:.2f}-{hl[1]:.2f} | boot {bt[0]:.2f}-{bt[1]:.2f})")
    pd.DataFrame(resumo).to_csv(os.path.join(a.out, "comparacao_grupos.csv"), index=False)

    # 2. Normalidade
    w("\n== Shapiro-Wilk (p) ==")
    total = nao_rej = 0
    for f in feats:
        for c in ("sem", "com"):
            for nome, G in ((ref, R), (alvo, A)):
                pv = est.shapiro(G[f"{f}_{c}"])
                total += 1
                nao_rej += bool(pv >= 0.05)
                w(f"{f:9s} {c} {nome}: p={pv:.4f} {'normal' if pv >= 0.05 else 'rejeita'}")
    w(f"Não rejeitadas: {nao_rej} de {total}; rejeitadas: {total - nao_rej}")

    # 3. Efeito da filtragem (pareado)
    w("\n== Wilcoxon pareado: sem vs com filtro (N total) ==")
    for f in feats:
        pv, rp = est.wilcoxon_pareado(df[f"{f}_sem"].values, df[f"{f}_com"].values)
        w(f"{f:9s}: p={pv:.4g} r_pareado={rp:.3f}")

    # 4. Covariável (ex.: idade)
    cv = a.covariavel
    if cv and cv in df.columns and df[cv].notna().all():
        w(f"\n== Covariável: {cv} ==")
        mr, ma = est.mediana_iq(R[cv].values), est.mediana_iq(A[cv].values)
        w(f"{ref} mediana {mr[0]:g} [{mr[1]:g}-{mr[2]:g}] | {alvo} mediana {ma[0]:g} "
          f"[{ma[1]:g}-{ma[2]:g}] | Mann-Whitney p={est.mann_whitney(R[cv].values, A[cv].values)[0]:.4f}")
        from scipy.stats import spearmanr
        rho, pv = spearmanr(df[cv], df[f"{principal}_sem"])
        w(f"Spearman {cv} x {principal}_sem: rho={rho:.3f} p={pv:.4f}")

    # 5. Contagem por categoria (ex.: sexo)
    if a.contagem and a.contagem in df.columns:
        w(f"\n== Contagem por {a.contagem} ==")
        w(df.groupby(["grupo", a.contagem]).size().to_string())

    # 6. Sensibilidade: remoção do(s) maior(es) valor(es) da característica principal no grupo alvo
    if a.remover_extremos > 0:
        k = a.remover_extremos
        w(f"\n== Sensibilidade: remoção dos {k} maiores {principal}_sem no grupo {alvo} ==")
        fora = A[f"{principal}_sem"].nlargest(k).index
        w(f"Removidos: {', '.join(fora)}")
        A2 = A.drop(fora)
        w(f"Mediana {alvo} sem extremos: {A2[f'{principal}_sem'].median():.6g}")
        for c in ("sem", "com"):
            col = f"{principal}_{c}"
            pv, rr = est.mann_whitney(R[col].values, A2[col].values)
            w(f"{col}: p={pv:.4f} r={rr:.3f} AUC={est.auc(R[col].values, A2[col].values):.3f}")
        w(f"Shapiro-Wilk {principal}_sem {alvo} sem extremos: p={est.shapiro(A2[f'{principal}_sem']):.4f}")

    # 7. Exclusão por comentário clínico (diagnósticos atípicos)
    cc = a.coluna_comentario
    if a.regex_atipicos and cc in df.columns:
        pat = re.compile(a.regex_atipicos, re.I)
        atip = A[A[cc].fillna("").astype(str).str.contains(pat)]
        w(f"\n== Exclusão de {alvo} com comentário clínico atípico (n = {len(atip)}) ==")
        if len(atip):
            w(atip[[c for c in (cv, cc) if c and c in atip.columns]].to_string())
            A3 = A.drop(atip.index)
            for f in feats:
                for c in ("sem", "com"):
                    col = f"{f}_{c}"
                    pv, rr = est.mann_whitney(R[col].values, A3[col].values)
                    w(f"{col:13s} n={len(R)}+{len(A3)} p={pv:.3e} r={rr:.3f} "
                      f"AUC={est.auc(R[col].values, A3[col].values):.3f}")

    # 8. Classificação LOOCV
    if a.loocv:
        w("\n== Regressão logística LOOCV (RobustScaler dentro da dobra) ==")
        y = (df.grupo == alvo).astype(int).values
        for c in ("sem", "com"):
            a1 = est.loocv_auc(df[[f"{principal}_{c}"]].values, y)
            txt = f"{c}: {principal} isolada AUC={a1:.3f}"
            if len(feats) > 1:
                a3 = est.loocv_auc(df[[f"{f}_{c}" for f in feats]].values, y)
                txt += f" | {'+'.join(feats)} AUC={a3:.3f}"
            w(txt)

    if not a.sem_figuras:
        figuras.boxplots(df, [ref, alvo], rotulos, feats, pvals, a.out, a.rotulo_filtro)

    texto = "\n".join(L)
    open(os.path.join(a.out, "relatorio.txt"), "w", encoding="utf-8").write(texto)
    print(texto)


# -------------------------------------------------------------------- parser
def _parser():
    ap = argparse.ArgumentParser(prog="python -m tremor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    D = argparse.ArgumentDefaultsHelpFormatter

    def comum_arquivo(p):
        p.add_argument("--padrao-arquivo", default="{id}_{tarefa}_{punho}Wrist.txt",
                       help="nome do arquivo de sinal; campos {id}, {tarefa}, {punho}")
        p.add_argument("--tarefa", default="Relaxed")
        p.add_argument("--punho", default="Right", help="Right ou Left")
        p.add_argument("--largura-id", type=int, default=3, help="zeros à esquerda no id")
        p.add_argument("--config", help="arquivo JSON com opções")

    # amostra
    p = sub.add_parser("amostra", help="montar tabela de indivíduos", formatter_class=D)
    p.add_argument("--pacientes", required=False, default="patients",
                   help="pasta com os metadados JSON (um por indivíduo)")
    p.add_argument("--padrao-paciente", default="patient_*.json")
    p.add_argument("--campo-condicao", default="condition")
    p.add_argument("--grupos", nargs="+", default=["Healthy=Saudavel", "Parkinson's=Parkinson"],
                   help="condição_no_JSON=rótulo_do_grupo; condições não listadas são ignoradas")
    p.add_argument("--raw", help="se dada, remove indivíduos sem arquivo de sinal nesta pasta")
    p.add_argument("--ids", help="lista fixa de ids a incluir, separados por vírgula")
    p.add_argument("--excluir-ids", help="ids a excluir, separados por vírgula")
    p.add_argument("--n-por-grupo", type=int, default=0, help="0 = todos")
    p.add_argument("--balancear", action="store_true", help="mesmo n em todos os grupos (o do menor)")
    p.add_argument("--selecao", choices=["primeiros", "aleatoria"], default="primeiros",
                   help="como escolher os n por grupo")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--saida", default="amostra.csv")
    comum_arquivo(p)
    p.set_defaults(func=cmd_amostra)

    # extrair
    p = sub.add_parser("extrair", help="calcular características dos sinais", formatter_class=D)
    p.add_argument("--raw", default="dados_brutos", help="pasta com os arquivos de sinal")
    p.add_argument("--amostra", default="amostra.csv", help="CSV com colunas id, grupo (+ opcionais)")
    p.add_argument("--out", default="saida")
    p.add_argument("--fs", type=float, default=100.0, help="taxa de amostragem (Hz)")
    p.add_argument("--descarte", type=int, default=48, help="amostras iniciais descartadas")
    p.add_argument("--sensor", choices=list(SENSORES), default="giroscopio")
    p.add_argument("--colunas", help="índices das colunas (ex.: 4,5,6); sobrepõe --sensor")
    p.add_argument("--combinacao", choices=["magnitude", "x", "y", "z"], default="magnitude")
    p.add_argument("--delimitador", default=",")
    p.add_argument("--banda", default="1,10", help="corte(s) do Butterworth em Hz")
    p.add_argument("--ordem", type=int, default=4)
    p.add_argument("--tipo-filtro", choices=["bandpass", "lowpass", "highpass", "bandstop"],
                   default="bandpass")
    p.add_argument("--tukey-alpha", type=float, default=0.2, help="janela da FMED")
    p.add_argument("--caracteristicas", default=",".join(CARACTERISTICAS),
                   help=f"disponíveis: {', '.join(CARACTERISTICAS)}")
    p.add_argument("--exemplos", help="ids da figura de sinais (2 por grupo); padrão: sorteio")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--rotulos", nargs="*", default=["Saudavel=Saudável"])
    p.add_argument("--sem-figuras", action="store_true")
    comum_arquivo(p)
    p.set_defaults(func=cmd_extrair)

    # analisar
    p = sub.add_parser("analisar", help="estatística e figuras", formatter_class=D)
    p.add_argument("--res", default="saida/resultados_por_individuo.csv")
    p.add_argument("--out", default="saida")
    p.add_argument("--ref", default="Saudavel", help="grupo de referência")
    p.add_argument("--alvo", default="Parkinson", help="grupo de interesse")
    p.add_argument("--caracteristicas", help="padrão: todas as colunas *_sem do CSV")
    p.add_argument("--principal", default="MAVSD",
                   help="característica usada em covariável, sensibilidade e LOOCV")
    p.add_argument("--covariavel", default="idade")
    p.add_argument("--contagem", default="sexo")
    p.add_argument("--remover-extremos", type=int, default=1,
                   help="n maiores valores do grupo alvo removidos na sensibilidade (0 desliga)")
    p.add_argument("--coluna-comentario", default="comentario_clinico")
    p.add_argument("--regex-atipicos", default=REGEX_ATIPICOS_PADS,
                   help='expressão para casos atípicos no grupo alvo ("" desliga)')
    p.add_argument("--bootstrap", type=int, default=5000, help="reamostragens da AUC (0 desliga)")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--loocv", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--rotulos", nargs="*", default=["Saudavel=Saudável"])
    p.add_argument("--rotulo-filtro", default="1–10 Hz", help="texto do painel com filtro")
    p.add_argument("--sem-figuras", action="store_true")
    p.add_argument("--config", help="arquivo JSON com opções")
    p.set_defaults(func=cmd_analisar)
    return ap


def main(argv=None):
    ap = _parser()
    a = ap.parse_args(argv)
    if getattr(a, "config", None):
        conf = json.load(open(a.config, encoding="utf-8"))
        sub = ap._subparsers._group_actions[0].choices[a.cmd]
        sub.set_defaults(**{k.replace("-", "_"): v for k, v in conf.items()})
        a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
