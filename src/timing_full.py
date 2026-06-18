"""Mede tempo real de 1 run completa do AG (pop=100, gen=50)."""
import time
from pathlib import Path
from sklearn.model_selection import train_test_split

from .data_loader import load_adult, SENSITIVE_COL
from .baseline_greedy import greedy_dm_anonymize
from .ag_task import run_ag_task, AGConfig

ROOT = Path(__file__).resolve().parents[1]


def main():
    df = load_adult(ROOT / "data")
    df_dev, df_test = train_test_split(df, test_size=0.2, random_state=42,
                                       stratify=df[SENSITIVE_COL])
    df_train, df_val = train_test_split(df_dev, test_size=0.3, random_state=42,
                                        stratify=df_dev[SENSITIVE_COL])
    print(f"train={len(df_train)}, val={len(df_val)}, test={len(df_test)}")

    for k in [10, 100]:
        print(f"\n=== k={k} ===")
        t0 = time.time()
        chrom_g, df_g, dm_g = greedy_dm_anonymize(df_train, k)
        t_g = time.time() - t0
        print(f"Greedy: {chrom_g} | DM={dm_g:.0f} | {t_g:.2f}s")

        cfg = AGConfig(pop_size=100, n_gen=50, seed=1)
        t0 = time.time()
        chrom_ag, fit_ag, hist = run_ag_task(df_train, df_val, k, cfg,
                                              seed_chromosome=chrom_g)
        t_ag = time.time() - t0
        print(f"AG full: {chrom_ag} | fit={fit_ag:.4f} | {t_ag:.1f}s = {t_ag/60:.2f}min")
        print(f"Hist (10 ultimas): {[round(h,4) for h in hist[-10:]]}")


if __name__ == "__main__":
    main()
