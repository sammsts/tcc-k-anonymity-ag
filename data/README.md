# Adult Dataset

O Adult Dataset (UCI ML Repository) não é incluído neste repositório por questões de tamanho e licenciamento. Baixe os arquivos diretamente:

```bash
curl -sSL -o adult.data https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
curl -sSL -o adult.test https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
```

Ambos os arquivos devem ficar neste diretório (`data/`).

## Sobre o dataset

- **Origem**: UCI Machine Learning Repository
- **Registros brutos**: 48.842 (treino + teste)
- **Registros após limpeza** (drop NaN nos QIs e no atributo sensível): **45.222**
- **Atributos**: 14 (8 QIs, 1 sensível, 4 não-QI)
- **Atributo-alvo**: `income` (renda > 50K ou ≤ 50K por ano)

A limpeza é feita automaticamente em `src/data_loader.py::load_adult()`.

## Referência

Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
