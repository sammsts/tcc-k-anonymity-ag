"""Sanity check + estimativa de tempo."""
import time
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .data_loader import load_adult, QI_COLS, SENSITIVE_COL
from .vghs import apply_generalization, total_search_space, QI_ORDER
from .k_anonymity import (
    verificar_k_anonimato, discernibility_metric, discernibility_metric_normalized
)
from .baseline_greedy import greedy_dm_anonymize
from .ag_task import run_ag_task, AGConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    print("=" * 60)
    print("SANITY CHECK + TIMING")
    print("=" * 60)

    t0 = time.time()
    df = load_adult(ROOT / "data")
    print(f"\n[1] Dataset carregado: {len(df)} registros, {df.shape[1]} cols")
    print(f"    Tempo carga: {time.time()-t0:.2f}s")
    print(f"    Distribuicao income: {df[SENSITIVE_COL].value_counts(normalize=True).to_dict()}")
    print(f"    Espaco de busca total (VGHs): {total_search_space()}")

    df_dev, df_test = train_test_split(df, test_size=0.2, random_state=42,
                                       stratify=df[SENSITIVE_COL])
    df_train, df_val = train_test_split(df_dev, test_size=0.3, random_state=42,
                                        stratify=df_dev[SENSITIVE_COL])
    print(f"\n[2] Splits: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    k = 10
    print(f"\n[3] Sanity: cromossomo zero (sem generalizar) com k={k}")
    df0 = apply_generalization(df_train, [0] * 8)
    print(f"    k-anon valido? {verificar_k_anonimato(df0, k)}")

    df_full = apply_generalization(df_train, [3, 2, 2, 2, 2, 1, 1, 2])
    print(f"\n[4] Sanity: cromossomo maximo (supressao total) com k={k}")
    print(f"    k-anon valido? {verificar_k_anonimato(df_full, k)}")
    print(f"    DM = {discernibility_metric(df_full, k):.0f}")

    print(f"\n[5] Rodando baseline guloso k={k}...")
    t0 = time.time()
    chrom_greedy, df_anon_greedy, dm_greedy = greedy_dm_anonymize(df_train, k)
    t_greedy = time.time() - t0
    print(f"    Cromossomo: {chrom_greedy}")
    print(f"    DM: {dm_greedy:.0f} (norm: {discernibility_metric_normalized(df_anon_greedy, k):.4f})")
    print(f"    Tempo: {t_greedy:.2f}s")

    print(f"\n[6] Rodando AG MINI (pop=20, gen=10) k={k} para timing...")
    cfg_mini = AGConfig(pop_size=20, n_gen=10, seed=1)
    t0 = time.time()
    chrom_ag, fit_ag, hist = run_ag_task(df_train, df_val, k, cfg_mini)
    t_ag_mini = time.time() - t0
    print(f"    Cromossomo: {chrom_ag}")
    print(f"    Fitness (acuracia proxy): {fit_ag:.4f}")
    print(f"    Hist (5 ultimas geracoes): {[round(h,4) for h in hist[-5:]]}")
    print(f"    Tempo: {t_ag_mini:.2f}s")

    n_eval_mini = 20 + 10 * 20
    n_eval_full = 100 + 50 * 100
    estimativa = t_ag_mini * (n_eval_full / n_eval_mini)
    print(f"\n[7] Estimativa de 1 run AG full (pop=100, gen=50):")
    print(f"    ~{n_eval_full} avaliacoes vs {n_eval_mini} no mini")
    print(f"    Estimativa upper bound (sem cache): {estimativa:.1f}s = {estimativa/60:.2f}min")
    print(f"    Para 50 runs (5 k x 10): {estimativa*50/60:.1f}min = {estimativa*50/3600:.1f}h")

    print("\n" + "=" * 60)
    print("Sanity OK. Pronto para experimento completo.")
    print("=" * 60)


if __name__ == "__main__":
    main()
