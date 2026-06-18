"""Baseline guloso (Greedy-DM) inspirado em Datafly/Bayardo-Agrawal.

A cada iteração escolhe o QI cujo próximo nível de generalização minimiza
o aumento da Discernibility Metric (DM); itera até satisfazer k."""
from __future__ import annotations
import pandas as pd
from .vghs import QI_ORDER, VGH_LEVELS, apply_generalization
from .k_anonymity import discernibility_metric, verificar_k_anonimato


def greedy_dm_anonymize(df: pd.DataFrame, k: int) -> tuple[list[int], pd.DataFrame, float]:
    """Retorna (cromossomo, df_anon, dm) da solução gulosa que satisfaz k."""
    chromosome = [0] * len(QI_ORDER)
    max_levels = [VGH_LEVELS[c] - 1 for c in QI_ORDER]

    df_anon = apply_generalization(df, chromosome)
    while not verificar_k_anonimato(df_anon, k):
        best_idx = None
        best_dm = float("inf")
        for i in range(len(chromosome)):
            if chromosome[i] >= max_levels[i]:
                continue
            cand = chromosome.copy()
            cand[i] += 1
            cand_anon = apply_generalization(df, cand)
            dm = discernibility_metric(cand_anon, k)
            if dm < best_dm:
                best_dm = dm
                best_idx = i
        if best_idx is None:
            break
        chromosome[best_idx] += 1
        df_anon = apply_generalization(df, chromosome)

    return chromosome, df_anon, discernibility_metric(df_anon, k)
