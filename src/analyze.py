"""Análise estatística dos resultados: médias, desvios, Wilcoxon."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, ranksums

ROOT = Path(__file__).resolve().parents[1]


def load_results(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby("k"):
        acc_g = g["greedy_acc"].values
        acc_a = g["ag_acc"].values
        f1_g = g["greedy_f1_macro"].values
        f1_a = g["ag_f1_macro"].values
        dm_g = g["greedy_dm_norm"].values
        dm_a = g["ag_dm_norm"].values
        t_g_all = g["greedy_time_s"].values
        t_a_all = g["ag_time_s"].values
        time_filter = (t_a_all < 3600) & (t_g_all < 3600)
        t_g = t_g_all[time_filter]
        t_a = t_a_all[time_filter]
        if len(t_a) == 0:
            t_a = t_a_all
            t_g = t_g_all

        try:
            w_acc = ranksums(acc_a, acc_g, alternative="two-sided").pvalue
        except Exception:
            w_acc = float("nan")
        try:
            w_f1 = ranksums(f1_a, f1_g, alternative="two-sided").pvalue
        except Exception:
            w_f1 = float("nan")

        rows.append({
            "k": int(k),
            "n_runs": len(g),
            "acc_ag_mean": acc_a.mean(),
            "acc_ag_std": acc_a.std(ddof=1) if len(acc_a) > 1 else 0.0,
            "acc_g_mean": acc_g.mean(),
            "acc_g_std": acc_g.std(ddof=1) if len(acc_g) > 1 else 0.0,
            "acc_delta_pp": (acc_a.mean() - acc_g.mean()) * 100,
            "acc_pvalue": w_acc,
            "f1_ag_mean": f1_a.mean(),
            "f1_ag_std": f1_a.std(ddof=1) if len(f1_a) > 1 else 0.0,
            "f1_g_mean": f1_g.mean(),
            "f1_g_std": f1_g.std(ddof=1) if len(f1_g) > 1 else 0.0,
            "f1_delta": f1_a.mean() - f1_g.mean(),
            "f1_pvalue": w_f1,
            "dm_ag_mean": dm_a.mean(),
            "dm_g_mean": dm_g.mean(),
            "dm_ag_vs_g_pct": (dm_a.mean() - dm_g.mean()) / dm_g.mean() * 100 if dm_g.mean() > 0 else float("nan"),
            "time_ag_mean_s": t_a.mean(),
            "time_ag_std_s": t_a.std(ddof=1) if len(t_a) > 1 else 0.0,
            "time_g_mean_s": t_g.mean(),
            "time_ratio": t_a.mean() / t_g.mean() if t_g.mean() > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("k").reset_index(drop=True)


def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "resultados" / "results.jsonl"
    df = load_results(results_path)
    print(f"Carregados {len(df)} registros de {results_path}\n")
    print(f"k values: {sorted(df['k'].unique())}")
    print(f"runs per k: {df.groupby('k').size().to_dict()}\n")

    summary = summarize(df)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 200)
    print(summary.to_string(index=False))

    out_csv = results_path.parent / "summary.csv"
    summary.to_csv(out_csv, index=False, float_format="%.6f")
    print(f"\nSummary CSV: {out_csv}")

    orig_path = results_path.parent / "original_baseline.json"
    if orig_path.exists():
        with open(orig_path) as fh:
            orig = json.load(fh)
        print(f"\nOriginal (sem anonimizacao):")
        print(f"  acuracia: {orig['accuracy']:.4f}")
        print(f"  f1_macro: {orig['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
