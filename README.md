# Análise de tremor por sensores inerciais de smartwatch

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Licença](https://img.shields.io/badge/licen%C3%A7a-MIT-green)
![Base](https://img.shields.io/badge/dados-PADS%20%2F%20PhysioNet-orange)

Pipeline em Python para extrair características de sinais de giroscópio ou acelerômetro e comparar dois grupos de indivíduos (por exemplo, Doença de Parkinson vs. controles saudáveis). Foi desenvolvido para o artigo **"Discriminação entre indivíduos com Doença de Parkinson e controles saudáveis por meio da variabilidade do sinal de giroscópio em repouso"** (X SABIO, UFPE), com dados da base pública [PADS – Parkinson's Disease Smartwatch dataset](https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/) (PhysioNet, DOI 10.13026/m0w9-zx22).

Os valores padrão reproduzem a análise do artigo. Todos os parâmetros (tarefa, punho, sensor, eixo, filtro, características, grupos, tamanho da amostra) podem ser trocados por linha de comando ou por um arquivo JSON, sem mexer no código.

## Instalação

Python 3.9 ou superior. Na pasta do repositório:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
```

Isso instala o comando `tremor`. Os exemplos abaixo usam `python -m tremor`, que funciona com ou sem a instalação (basta `pip install -r requirements.txt`).

## Dados

Os dados não fazem parte do repositório. Baixe a base PADS no PhysioNet e aponte o pipeline para:

- a pasta dos **metadados** (`patients/patient_NNN.json`, com `condition`, `age`, `gender`, `disease_comment`);
- a pasta dos **sinais** (`movement/timeseries/`), com um arquivo de texto por indivíduo, tarefa e punho.

Cada arquivo de sinal tem 7 colunas separadas por vírgula, amostradas a 100 Hz:

| 0 | 1–3 | 4–6 |
|---|---|---|
| tempo | acelerômetro x/y/z | giroscópio x/y/z (rad/s) |

Tarefas disponíveis no nome dos arquivos (`--tarefa`): `Relaxed`, `RelaxedTask`, `StretchHold`, `HoldWeight`, `DrinkGlas`, `CrossArms`, `TouchNose`, `Entrainment`. Punhos (`--punho`): `Left` ou `Right`.

Outras bases funcionam desde que tenham um arquivo por indivíduo e uma tabela com `id` e `grupo` (veja `--colunas`, `--delimitador` e `--padrao-arquivo`).

## Uso

São três passos independentes:

```bash
# 1. Tabela de indivíduos (id, grupo, idade, sexo, arquivo, comentário clínico)
python -m tremor amostra --pacientes patients --raw movement/timeseries --saida amostra.csv

# 2. Características de cada sinal, sem e com filtro
python -m tremor extrair --raw movement/timeseries --amostra amostra.csv --out saida

# 3. Estatística e figuras
python -m tremor analisar --res saida/resultados_por_individuo.csv --out saida
```

`python -m tremor <passo> -h` lista todas as opções com os valores padrão.

### Reproduzir o artigo

**N = 158** (todos os 79 controles e os 79 primeiros Parkinson por ID):

```bash
python -m tremor amostra --pacientes patients --raw movement/timeseries --balancear --selecao primeiros --saida amostra_n158.csv
python -m tremor extrair --raw movement/timeseries --amostra amostra_n158.csv --out saida158 --config exemplos/config_artigo.json
python -m tremor analisar --res saida158/resultados_por_individuo.csv --out saida158
```

**N = 32** (amostra da primeira versão):

```bash
python -m tremor amostra --pacientes patients --raw movement/timeseries --saida amostra_n32.csv \
  --ids 1,3,21,27,34,45,46,52,54,62,78,80,81,84,97,101,4,5,6,8,9,10,12,13,14,19,20,22,25,26,28,29
python -m tremor extrair --raw movement/timeseries --amostra amostra_n32.csv --out saida32 --exemplos 003,081,012,029
python -m tremor analisar --res saida32/resultados_por_individuo.csv --out saida32
```

### Outras análises

```bash
# Punho esquerdo, outra tarefa
python -m tremor amostra --pacientes patients --raw movement/timeseries --punho Left --tarefa CrossArms --saida amostra.csv

# Acelerômetro, só o eixo z, passa-baixa de 15 Hz
python -m tremor extrair ... --sensor acelerometro --combinacao z --tipo-filtro lowpass --banda 15

# 30 indivíduos por grupo sorteados (semente fixa)
python -m tremor amostra ... --n-por-grupo 30 --selecao aleatoria --semente 7

# Comparar controles com "Other Movement Disorders"
python -m tremor amostra ... --grupos "Healthy=Saudavel" "Other Movement Disorders=Outros"
python -m tremor analisar ... --alvo Outros

# Tudo a partir de um arquivo
python -m tremor extrair --config exemplos/config_acelerometro_punho_esquerdo.json --raw ... --amostra ...
```

## Pipeline

1. **Leitura e combinação de eixos.** Seleciona as colunas do sensor e calcula a magnitude vetorial $\|\omega\| = \sqrt{\omega_x^2 + \omega_y^2 + \omega_z^2}$, ou usa um eixo isolado (`--combinacao`).
2. **Descarte inicial.** Remove as primeiras 48 amostras (0,48 s), onde aparece a vibração de notificação do relógio (`--descarte`).
3. **Filtragem.** Butterworth de 4ª ordem, passa-faixa 1–10 Hz, aplicado com `filtfilt` (fase zero, ordem efetiva dobrada). Cada característica é calculada no sinal sem filtro e no filtrado.
4. **Características** (`tremor/caracteristicas.py`):
   - **MAVSD**: média do valor absoluto da segunda diferença, $\frac{1}{N-2}\sum |x_i - 2x_{i-1} + x_{i-2}|$.
   - **FMED**: frequência mediana do espectro. Tendência linear removida, janela de Tukey ($\alpha = 0{,}2$), FFT de segmento único, energia $|X|^2$; é a primeira frequência em que a energia acumulada atinge 50%.
   - **Kurtosis**: curtose de Pearson (normal = 3).
5. **Estatística** (`tremor/estatistica.py`), entre o grupo de referência (`--ref`) e o grupo alvo (`--alvo`):
   - mediana e quartis; Shapiro-Wilk por grupo;
   - Mann-Whitney bilateral com tamanho de efeito r rank-biserial;
   - AUC com IC95% de Hanley-McNeil e por bootstrap estratificado (5000 reamostragens, semente 42);
   - Wilcoxon pareado sem vs. com filtro, com r pareado;
   - covariável (idade): Mann-Whitney entre grupos e Spearman com a característica principal;
   - sensibilidade: remoção dos maiores valores da característica principal no grupo alvo;
   - exclusão de casos com comentário clínico atípico (tremor essencial, parkinsonismo atípico, MSA, PSP etc.);
   - regressão logística com validação leave-one-out, característica principal isolada vs. todas juntas.

## Saídas

| Arquivo | Conteúdo |
|---|---|
| `resultados_por_individuo.csv` | características sem/com filtro de cada indivíduo |
| `parametros_extracao.json` | parâmetros usados na extração |
| `sinais_exemplos.png` | dois sinais por grupo |
| `comparacao_grupos.csv` | estatística por característica (mediana, p, r, AUC, ICs) |
| `relatorio.txt` | todos os testes em texto |
| `boxplot_<caracteristica>.png` | boxplots sem e com filtro |

## Acrescentar uma característica

Escreva uma função `f(x, cfg) -> float` em `tremor/caracteristicas.py` e registre-a no dicionário `CARACTERISTICAS`. Ela passa a ser aceita em `--caracteristicas`, entra no CSV e em toda a estatística.

```python
def rms(x, cfg=None):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))

CARACTERISTICAS["RMS"] = rms
```

## Testes

```bash
pip install -e ".[dev]"
pytest
```

Os testes geram uma mini base sintética no formato PADS e conferem as características (valores conhecidos), a estatística e o pipeline completo. Eles rodam automaticamente no GitHub a cada envio (aba **Actions**).

## Estrutura

```
tremor/
  sinais.py           leitura, combinação de eixos, descarte, filtro
  caracteristicas.py  MAVSD, FMED, Kurtosis (registro extensível)
  estatistica.py      testes, tamanhos de efeito, AUC, LOOCV
  figuras.py          boxplots e exemplos de sinais
  cli.py              comandos amostra / extrair / analisar
tests/                testes automáticos (pytest)
exemplos/             arquivos de configuração
.github/workflows/    integração contínua
```

## Autores

- Arianne Viana Alves da Silva — Departamento de Engenharia Biomédica, Universidade Federal de Pernambuco
- Arthur Coelho Bastos — Faculdade de Engenharia Elétrica, Universidade Federal de Uberlândia
- Leonardo Bongiovani — Faculdade de Engenharia Elétrica, Universidade Federal de Uberlândia
- Letícia Marinho — Faculdade de Engenharia Elétrica, Universidade Federal de Uberlândia
- Melyssa Júlia Neres Faria — Faculdade de Engenharia Elétrica, Universidade Federal de Uberlândia

## Como citar

O botão **Cite this repository**, no GitHub, gera a referência a partir do `CITATION.cff`. Cite também a base PADS:

> Varghese, J., Brenner, A., Plagwitz, L., van Alen, C., Fujarski, M., & Warnecke, T. (2024). *PADS – Parkinsons Disease Smartwatch dataset* (version 1.0.0). PhysioNet. https://doi.org/10.13026/m0w9-zx22
>
> Varghese, J., Brenner, A., Fujarski, M., van Alen, C. M., Plagwitz, L., & Warnecke, T. (2024). Machine learning in the Parkinson's disease smartwatch (PADS) dataset. *npj Parkinson's Disease*, 10, 9.

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).
