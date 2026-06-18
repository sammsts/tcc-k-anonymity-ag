"""Algoritmo Genético com fitness task-based (acurácia de ML) via DEAP.

Cromossomo: vetor de 8 inteiros indicando o nível VGH para cada QI.
Fitness: acurácia de um DecisionTreeClassifier (proxy rápido) ou 0.0 caso k violado.
Cache de avaliações é usado para evitar retraino de cromossomos repetidos.
Operador de reparo garante que indivíduos inválidos sejam generalizados
incrementalmente até satisfazerem k (técnica padrão de AG com restrições rígidas)."""
from __future__ import annotations
import random
from dataclasses import dataclass
import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from .vghs import QI_ORDER, VGH_LEVELS, apply_generalization
from .k_anonymity import verificar_k_anonimato, discernibility_metric

if not hasattr(creator, "FitnessMaxAG"):
    creator.create("FitnessMaxAG", base.Fitness, weights=(1.0,))
if not hasattr(creator, "IndividualAG"):
    creator.create("IndividualAG", list, fitness=creator.FitnessMaxAG)


@dataclass
class AGConfig:
    pop_size: int = 100
    n_gen: int = 50
    cx_prob: float = 0.8
    mut_prob_ind: float = 0.2
    mut_prob_gene: float = 0.1
    tournament_size: int = 3
    elite_frac: float = 0.05
    seed: int = 42


def _encode_features(df: pd.DataFrame, ref_columns=None):
    extra = [c for c in ["fnlwgt", "capital_gain", "capital_loss", "hours_per_week"] if c in df.columns]
    X = pd.get_dummies(df[QI_ORDER + extra].copy(), columns=QI_ORDER, drop_first=False)
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)
    return X


def _aptitude_factory(df_train_full: pd.DataFrame, df_val: pd.DataFrame, k: int, cache: dict):
    y_train_full = df_train_full["income"].values
    y_val = df_val["income"].values

    def evaluate(individual):
        key = tuple(individual)
        if key in cache:
            return (cache[key],)

        df_anon = apply_generalization(df_train_full, list(individual))
        if not verificar_k_anonimato(df_anon, k):
            cache[key] = 0.0
            return (0.0,)

        X_train = _encode_features(df_anon)
        X_val = _encode_features(df_val, ref_columns=X_train.columns)

        clf = DecisionTreeClassifier(max_depth=10, random_state=42)
        clf.fit(X_train.values, y_train_full)
        y_pred = clf.predict(X_val.values)
        acc = float(accuracy_score(y_val, y_pred))
        cache[key] = acc
        return (acc,)

    return evaluate


def _repair(individual, df: pd.DataFrame, k: int, max_levels: list[int],
            valid_cache: set) -> list[int]:
    """Se inválido, escolhe o QI cuja generalização +1 minimiza o aumento de DM,
    até virar válido (loop similar ao greedy mas local ao indivíduo)."""
    key = tuple(individual)
    if key in valid_cache:
        return individual
    df_anon = apply_generalization(df, list(individual))
    if verificar_k_anonimato(df_anon, k):
        valid_cache.add(tuple(individual))
        return individual

    while not verificar_k_anonimato(df_anon, k):
        best_idx = None
        best_dm = float("inf")
        for i in range(len(individual)):
            if individual[i] >= max_levels[i]:
                continue
            cand = list(individual)
            cand[i] += 1
            cand_anon = apply_generalization(df, cand)
            dm = discernibility_metric(cand_anon, k)
            if dm < best_dm:
                best_dm = dm
                best_idx = i
        if best_idx is None:
            break
        individual[best_idx] += 1
        df_anon = apply_generalization(df, list(individual))
    valid_cache.add(tuple(individual))
    return individual


def run_ag_task(df_train: pd.DataFrame, df_val: pd.DataFrame,
                k: int, cfg: AGConfig,
                seed_chromosome: list[int] | None = None
                ) -> tuple[list[int], float, list[float]]:
    """Roda o AG e retorna (melhor_cromossomo, melhor_fitness, hist_melhor_por_geracao).
    `seed_chromosome` (ex.: solução do greedy) é injetado na população inicial."""
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    toolbox = base.Toolbox()
    n_qi = len(QI_ORDER)
    max_levels = [VGH_LEVELS[c] - 1 for c in QI_ORDER]

    def init_gene(i):
        return random.randint(0, max_levels[i])

    def init_individual():
        return creator.IndividualAG([init_gene(i) for i in range(n_qi)])

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)

    def mutate_uniform(ind):
        for i in range(n_qi):
            if random.random() < cfg.mut_prob_gene:
                ind[i] = random.randint(0, max_levels[i])
        return (ind,)

    toolbox.register("mutate", mutate_uniform)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

    eval_cache: dict = {}
    valid_cache: set = set()
    toolbox.register("evaluate", _aptitude_factory(df_train, df_val, k, eval_cache))

    pop = toolbox.population(n=cfg.pop_size)
    if seed_chromosome is not None:
        pop[0] = creator.IndividualAG(list(seed_chromosome))
        for j in range(1, min(5, cfg.pop_size)):
            perturbed = list(seed_chromosome)
            idx = random.randint(0, n_qi - 1)
            perturbed[idx] = max(0, min(max_levels[idx], perturbed[idx] + random.choice([-1, 1])))
            pop[j] = creator.IndividualAG(perturbed)

    for ind in pop:
        _repair(ind, df_train, k, max_levels, valid_cache)
        ind.fitness.values = toolbox.evaluate(ind)

    elite_n = max(1, int(cfg.elite_frac * cfg.pop_size))
    hist_best = []

    for gen in range(cfg.n_gen):
        elites = [creator.IndividualAG(list(ind)) for ind in tools.selBest(pop, elite_n)]
        for e, src in zip(elites, tools.selBest(pop, elite_n)):
            e.fitness.values = src.fitness.values

        offspring = toolbox.select(pop, len(pop) - elite_n)
        offspring = [creator.IndividualAG(list(ind)) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cfg.cx_prob:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mut in offspring:
            if random.random() < cfg.mut_prob_ind:
                toolbox.mutate(mut)
                if mut.fitness.valid:
                    del mut.fitness.values

        for ind in offspring:
            if not ind.fitness.valid:
                _repair(ind, df_train, k, max_levels, valid_cache)
                ind.fitness.values = toolbox.evaluate(ind)

        pop[:] = elites + offspring
        best = tools.selBest(pop, 1)[0]
        hist_best.append(best.fitness.values[0])

    best = tools.selBest(pop, 1)[0]
    return list(best), float(best.fitness.values[0]), hist_best
