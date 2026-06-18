"""Gera as strings para substituição no LaTeX a partir do summary.csv."""
import json
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def fmt_pct(x: float) -> str:
    return f"{x*100:.1f}".replace(".", ",")


def fmt_pp(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.1f}".replace(".", ",")


def fmt_pvalue(p: float) -> str:
    if p < 0.001:
        return "$< 0,001$"
    return f"{p:.3f}".replace(".", ",")


def fmt_f1(x: float) -> str:
    return f"{x:.3f}".replace(".", ",")


def fmt_f1_pp(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.3f}".replace(".", ",")


def fmt_dm(x: float) -> str:
    return f"{x:.4f}".replace(".", ",")


def fmt_dm_pct(x: float) -> str:
    sign = "+" if x >= 0 else "-"
    return f"{sign}{abs(x):.1f}\\%"


def fmt_time(x: float) -> str:
    return f"{x:.0f}".replace(".", ",")


def fmt_time_decimal(x: float) -> str:
    return f"{x:.1f}".replace(".", ",")


def main():
    summary = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else ROOT / "resultados" / "summary.csv")
    orig_path = (ROOT / "resultados" / "original_baseline.json")
    with open(orig_path) as fh:
        orig = json.load(fh)

    print("=" * 70)
    print("VALORES PARA SUBSTITUICAO NO LaTeX")
    print("=" * 70)

    print(f"\n[Original RF (limite superior, sem anonimizacao)]")
    print(f"  acuracia = {fmt_pct(orig['accuracy'])}% ({orig['accuracy']*100:.2f}%)")
    print(f"  f1_macro = {fmt_f1(orig['f1_macro'])}")

    print(f"\n[Tabela 4: Acuracia do Random Forest]")
    print(f"  k  | AG (mean +/- std) | Greedy (mean +/- std) | delta_pp | p-valor")
    for _, r in summary.iterrows():
        k = int(r["k"])
        ag = f"{fmt_pct(r['acc_ag_mean'])} ($\\pm${r['acc_ag_std']*100:.1f})".replace(".", ",")
        gr = f"{fmt_pct(r['acc_g_mean'])} ($\\pm${r['acc_g_std']*100:.1f})".replace(".", ",")
        delta = fmt_pp(r["acc_delta_pp"]/100*100)  # already in pp
        delta = fmt_pp(r["acc_delta_pp"])
        p = fmt_pvalue(r["acc_pvalue"])
        print(f"  k={k}: AG={ag} | Greedy={gr} | delta={delta} pp | p={p}")

    print(f"\n[Tabela 5: F1 Score Macro]")
    for _, r in summary.iterrows():
        k = int(r["k"])
        ag = f"{fmt_f1(r['f1_ag_mean'])} ($\\pm${r['f1_ag_std']:.3f})".replace(".", ",")
        gr = f"{fmt_f1(r['f1_g_mean'])} ($\\pm${r['f1_g_std']:.3f})".replace(".", ",")
        delta = fmt_f1_pp(r["f1_delta"])
        p = fmt_pvalue(r["f1_pvalue"])
        print(f"  k={k}: AG={ag} | Greedy={gr} | delta={delta} | p={p}")

    print(f"\n[Tabela 6: Discernibility Metric Normalizada]")
    for _, r in summary.iterrows():
        k = int(r["k"])
        dm_ag = fmt_dm(r["dm_ag_mean"])
        dm_g = fmt_dm(r["dm_g_mean"])
        dm_pct = fmt_dm_pct(r["dm_ag_vs_g_pct"])
        delta_acc_pp = fmt_pp(r["acc_delta_pp"]) + " p.p."
        print(f"  k={k}: AG-DM={dm_ag} | Greedy-DM={dm_g} | AG-vs-G={dm_pct} | acuracia +={delta_acc_pp}")

    print(f"\n[Tabela 7: Tempos]")
    for _, r in summary.iterrows():
        k = int(r["k"])
        t_ag = f"{r['time_ag_mean_s']:.0f} $\\pm$ {r['time_ag_std_s']:.0f}"
        t_g = fmt_time_decimal(r['time_g_mean_s'])
        ratio = f"~{r['time_ratio']:.0f}x"
        print(f"  k={k}: AG={t_ag}s | Greedy={t_g}s | ratio={ratio}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
