"""Runner completo: k ∈ {5,10,20,50,100}, N execuções independentes.

Para cada (k, seed):
  - Greedy-DM: 1 execução (determinístico, mas reavaliamos RF com seed)
  - AG-Task: AG com seed distinta, depois avaliação final com RF
Métricas finais: acurácia + F1 do RandomForest no conjunto de teste,
DM normalizada, tempo de execução."""
import argparse
import json
import time
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import load_adult, SENSITIVE_COL
from .vghs import apply_generalization
from .k_anonymity import (
    discernibility_metric, discernibility_metric_normalized, verificar_k_anonimato,
)
from .baseline_greedy import greedy_dm_anonymize
from .ag_task import run_ag_task, AGConfig
from .evaluator import evaluate_final, evaluate_original

ROOT = Path(__file__).resolve().parents[1]


def run_single(df_train, df_val, df_test, k, seed, greedy_chrom):
    """Roda greedy + AG para um (k, seed) e retorna métricas finais."""
    t0 = time.time()
    t_greedy_anon = 0.0  # já computado fora

    eval_g = evaluate_final(greedy_chrom, df_train, df_test, seed=seed)
    df_train_anon_g = apply_generalization(df_train, greedy_chrom)
    dm_g = discernibility_metric(df_train_anon_g, k)
    dm_g_norm = discernibility_metric_normalized(df_train_anon_g, k)

    cfg = AGConfig(pop_size=100, n_gen=50, seed=seed)
    t0 = time.time()
    ag_chrom, ag_fit, hist = run_ag_task(df_train, df_val, k, cfg,
                                          seed_chromosome=greedy_chrom)
    t_ag = time.time() - t0
    eval_ag = evaluate_final(ag_chrom, df_train, df_test, seed=seed)
    df_train_anon_ag = apply_generalization(df_train, ag_chrom)
    dm_ag = discernibility_metric(df_train_anon_ag, k)
    dm_ag_norm = discernibility_metric_normalized(df_train_anon_ag, k)

    return {
        "k": k,
        "seed": seed,
        "greedy_chromosome": greedy_chrom,
        "greedy_acc": eval_g["accuracy"],
        "greedy_f1_macro": eval_g["f1_macro"],
        "greedy_f1_weighted": eval_g["f1_weighted"],
        "greedy_dm": dm_g,
        "greedy_dm_norm": dm_g_norm,
        "ag_chromosome": ag_chrom,
        "ag_acc": eval_ag["accuracy"],
        "ag_f1_macro": eval_ag["f1_macro"],
        "ag_f1_weighted": eval_ag["f1_weighted"],
        "ag_dm": dm_ag,
        "ag_dm_norm": dm_ag_norm,
        "ag_fit_proxy": ag_fit,
        "ag_time_s": t_ag,
        "ag_hist": hist,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", type=int, nargs="+",
                        default=[5, 10, 20, 50, 100])
    parser.add_argument("--n_runs", type=int, default=10)
    parser.add_argument("--out", type=str,
                        default=str(ROOT / "resultados" / "results.jsonl"))
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="adiciona ao seed base (1..n_runs)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Carregando dataset...", flush=True)
    df = load_adult(ROOT / "data")
    df_dev, df_test = train_test_split(df, test_size=0.2, random_state=42,
                                       stratify=df[SENSITIVE_COL])
    df_train, df_val = train_test_split(df_dev, test_size=0.3, random_state=42,
                                        stratify=df_dev[SENSITIVE_COL])

    print(f"Split: train={len(df_train)} val={len(df_val)} test={len(df_test)}", flush=True)

    print("Avaliando dataset original (referência sem anonimização)...", flush=True)
    eval_orig = evaluate_original(df_train, df_test, seed=42)
    with open(out_path.parent / "original_baseline.json", "w") as fh:
        json.dump(eval_orig, fh, indent=2)
    print(f"Original RF: acc={eval_orig['accuracy']:.4f} f1={eval_orig['f1_macro']:.4f}", flush=True)

    greedy_cache: dict[int, tuple] = {}
    greedy_times: dict[int, float] = {}
    for k in args.ks:
        print(f"Greedy k={k}...", flush=True)
        t0 = time.time()
        chrom_g, _, _ = greedy_dm_anonymize(df_train, k)
        t_g = time.time() - t0
        greedy_cache[k] = chrom_g
        greedy_times[k] = t_g
        print(f"  Greedy k={k}: {chrom_g} | t={t_g:.2f}s", flush=True)

    fh = open(out_path, "w")
    for k in args.ks:
        for run_i in range(1, args.n_runs + 1):
            seed = args.seed_offset + run_i
            t0 = time.time()
            print(f"[k={k} run={run_i}/{args.n_runs} seed={seed}] iniciando...", flush=True)
            res = run_single(df_train, df_val, df_test, k, seed, greedy_cache[k])
            res["greedy_time_s"] = greedy_times[k]
            res["wall_time_s"] = time.time() - t0
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            print(f"  greedy_acc={res['greedy_acc']:.4f} ag_acc={res['ag_acc']:.4f} "
                  f"delta={res['ag_acc']-res['greedy_acc']:+.4f} "
                  f"ag_time={res['ag_time_s']:.1f}s wall={res['wall_time_s']:.1f}s",
                  flush=True)
    fh.close()
    print(f"\nResultados em: {out_path}", flush=True)


if __name__ == "__main__":
    main()
