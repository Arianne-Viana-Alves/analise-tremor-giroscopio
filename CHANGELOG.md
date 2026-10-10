# Histórico de versões

## [1.0.0] - 2026-10-10
### Adicionado
- Pipeline em três comandos: `amostra`, `extrair`, `analisar`.
- Características MAVSD, FMED e Kurtosis, com registro para novas métricas.
- Filtro Butterworth configurável (tipo, banda, ordem), escolha de sensor, eixo, tarefa e punho.
- Estatística: Shapiro-Wilk, Mann-Whitney com r rank-biserial, AUC com IC95% (Hanley e bootstrap), Wilcoxon pareado, covariável, sensibilidade a extremos, exclusão de casos clínicos atípicos e regressão logística com LOOCV.
- Testes automáticos com dados sintéticos e integração contínua no GitHub Actions.
