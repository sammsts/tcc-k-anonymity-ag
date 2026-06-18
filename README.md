# TCC — Otimização da Utilidade de Dados Anonimizados via Algoritmo Genético Task-based

Implementação reproduzível do TCC **"Otimização da Utilidade de Dados Anonimizados com Suporte de Inteligência Artificial: Um Estudo Comparativo de Métricas e Abordagens"** (Samuel Maciel dos Santos, URI Santo Ângelo, 2026).

O projeto compara duas estratégias de anonimização sob *k-anonymity* no Adult Dataset:

1. **Greedy-DM** (baseline): heurística gulosa que minimiza a *Discernibility Metric* (Bayardo & Agrawal, 2005).
2. **AG-Task** (proposta): Algoritmo Genético via DEAP cuja função de aptidão é a acurácia de um classificador *Machine Learning* sobre os dados anonimizados (utilidade *task-based*).

## Resultados principais

| k | AG-Task (acc) | Greedy-DM (acc) | Δ (p.p.) | p-valor |
|---|---|---|---|---|
| 5 | 77,6% | 76,2% | **+1,4** | <0,001 |
| 10 | 78,5% | 76,2% | **+2,3** | <0,001 |
| 20 | 78,4% | 76,2% | **+2,2** | <0,001 |
| 50 | 74,8% | 75,0% | -0,2 | 0,001 |
| 100 | 74,8% | 75,0% | -0,2 | 0,001 |

50 execuções independentes (5 valores de k × 10 seeds), avaliação final com `RandomForestClassifier`, teste estatístico Wilcoxon rank-sum bilateral. Dados brutos em [resultados/results.jsonl](resultados/results.jsonl).

Veja a discussão completa no TCC: para `k ∈ {5, 10, 20}` o AG vence consistentemente (intervalo `+1,4` a `+2,3` p.p.); para `k ∈ {50, 100}` o espaço de cromossomos válidos colapsa e ambas as abordagens convergem para vizinhanças praticamente equivalentes.

## Estrutura do repositório

```
tcc-k-anonymity-ag/
├── src/
│   ├── data_loader.py       # Carga e limpeza do Adult Dataset
│   ├── vghs.py              # Hierarquias de Generalização (VGH)
│   ├── k_anonymity.py       # Verificação de k-anon e cálculo da DM
│   ├── baseline_greedy.py   # Algoritmo guloso (Greedy-DM)
│   ├── ag_task.py           # AG via DEAP + operador de reparo + cache
│   ├── evaluator.py         # Avaliação final com Random Forest
│   ├── run_experiment.py    # Runner do experimento completo
│   ├── analyze.py           # Análise estatística (médias, desvios, Wilcoxon)
│   ├── latex_subs.py        # Formata números para inserção no LaTeX
│   ├── sanity_check.py      # Sanity check rápido do pipeline
│   └── timing_full.py       # Medição de tempo de 1 run completa
├── data/                    # (vazio) baixe o dataset aqui — veja abaixo
├── resultados/
│   ├── results.jsonl        # Resultados brutos das 50 execuções
│   ├── summary.csv          # Estatísticas agregadas
│   ├── original_baseline.json  # Acurácia/F1 sobre dados não-anonimizados
│   └── run.log              # Log da execução completa
├── requirements.txt
└── README.md
```

## Pré-requisitos

- Python 3.11+ (testado em 3.13)
- ~16 GB RAM recomendado (8 GB suficiente)
- Sem GPU (DecisionTree e RandomForest não se beneficiam)

## Instalação

```bash
git clone https://github.com/samuelmdsantos/tcc-k-anonymity-ag.git
cd tcc-k-anonymity-ag
python -m venv .venv
source .venv/bin/activate    # Linux/Mac
.venv\Scripts\activate       # Windows PowerShell
pip install -r requirements.txt
```

## Obter o Adult Dataset

O Adult Dataset não é incluído no repositório. Baixe os dois arquivos do UCI ML Repository diretamente para `data/`:

```bash
curl -sSL -o data/adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -sSL -o data/adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

Após a limpeza (registros com `?` nos QIs ou no atributo sensível), o dataset efetivo é de **45.222 registros**.

## Reprodução do experimento

### Sanity check rápido (~30s)

Valida que o pipeline roda end-to-end com configuração reduzida do AG:

```bash
python -m src.sanity_check
```

### Experimento completo (~2 horas)

Roda 5 valores de `k` × 10 seeds = 50 execuções independentes:

```bash
python -m src.run_experiment --ks 5 10 20 50 100 --n_runs 10 --out resultados/results.jsonl
```

Argumentos:
- `--ks` — lista de valores de k a testar
- `--n_runs` — execuções independentes por k
- `--out` — caminho do arquivo JSONL de saída
- `--seed_offset` — offset para os seeds (útil para retomar execuções)

### Análise estatística

Gera `summary.csv` com médias, desvios padrão e p-valores (Wilcoxon rank-sum bilateral):

```bash
python -m src.analyze
```

### Formatar valores para o LaTeX

Imprime as strings formatadas (com vírgula como separador decimal) prontas para colar nas tabelas do TCC:

```bash
python -m src.latex_subs
```

## Configuração do AG

Hiperparâmetros padrão (em `src/ag_task.py`, classe `AGConfig`):

| Parâmetro | Valor |
|---|---|
| Tamanho da população | 100 |
| Número de gerações | 50 |
| Probabilidade de crossover | 0.8 |
| Probabilidade de mutação (indivíduo) | 0.2 |
| Probabilidade de mutação (gene) | 0.1 |
| Tamanho do torneio | 3 |
| Elitismo | top 5% |

Operadores especiais implementados:

- **Seed do greedy + perturbações** na população inicial (garante presença de cromossomos viáveis desde a geração 0).
- **Operador de reparo determinístico**: cromossomos que violam k são incrementados no QI cujo aumento minimiza o crescimento da DM, até a viabilidade ser restabelecida.
- **Cache de avaliações** entre gerações (evita retraino de cromossomos repetidos).

## Hierarquias de Generalização (VGH)

Implementadas em `src/vghs.py`:

| QI | Níveis | Hierarquia |
|---|---|---|
| `age` | 4 | exato → faixas 5 anos → faixas 10 anos → supressão |
| `workclass` | 3 | exato → Público/Privado/Autônomo → supressão |
| `education` | 3 | exato → Básico/Médio/Superior → supressão |
| `marital_status` | 3 | exato → Casado/Solteiro/Outro → supressão |
| `occupation` | 3 | exato → White/Blue-Collar/Serviços → supressão |
| `race` | 2 | exato → supressão |
| `sex` | 2 | exato → supressão |
| `native_country` | 3 | exato → continente → supressão |

Espaço de busca: 4×3×3×3×3×2×2×3 = **3.888 cromossomos**.

## Citação

Se você utilizar este código em trabalhos acadêmicos, cite:

```bibtex
@misc{santos2026tcc,
  author = {Samuel Maciel dos Santos},
  title = {Otimização da Utilidade de Dados Anonimizados com Suporte de Inteligência Artificial:
           Um Estudo Comparativo de Métricas e Abordagens},
  year = {2026},
  howpublished = {Trabalho de Conclusão de Curso, Ciência da Computação, URI Santo Ângelo},
  note = {Código-fonte disponível em \url{https://github.com/samuelmdsantos/tcc-k-anonymity-ag}}
}
```

## Licença

MIT — veja [LICENSE](LICENSE).

## Autor

Samuel Maciel dos Santos — `samuelmdsantos@aluno.santoangelo.uri.br`
Orientador: Prof. Alessandro Freitas de Oliveira

Universidade Regional Integrada do Alto Uruguai e das Missões (URI Santo Ângelo).
