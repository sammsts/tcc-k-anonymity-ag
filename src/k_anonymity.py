"""Verificação de k-anonymity e cálculo da Discernibility Metric (DM)."""
import pandas as pd
from .vghs import QI_ORDER


def verificar_k_anonimato(df_anon: pd.DataFrame, k: int,
                          qi_cols: list[str] = QI_ORDER) -> bool:
    """Verifica se cada grupo de equivalência tem tamanho >= k."""
    grupos = df_anon.groupby(qi_cols, observed=True).size()
    return bool((grupos >= k).all())


def group_sizes(df_anon: pd.DataFrame,
                qi_cols: list[str] = QI_ORDER) -> pd.Series:
    return df_anon.groupby(qi_cols, observed=True).size()


def discernibility_metric(df_anon: pd.DataFrame, k: int,
                          qi_cols: list[str] = QI_ORDER) -> float:
    """DM = soma sobre grupos de equivalência de |E_i|^2.
    Grupos que satisfazem k contribuem |E_i|^2; grupos suprimidos
    (|E_i| < k) contribuem |E_i| * N (penalidade clássica).
    Fonte: Bayardo e Agrawal (2005)."""
    sizes = group_sizes(df_anon, qi_cols)
    n = len(df_anon)
    valid_mask = sizes >= k
    dm_valid = float((sizes[valid_mask] ** 2).sum())
    dm_suppressed = float((sizes[~valid_mask] * n).sum())
    return dm_valid + dm_suppressed


def discernibility_metric_normalized(df_anon: pd.DataFrame, k: int,
                                     qi_cols: list[str] = QI_ORDER) -> float:
    """DM normalizada por n^2 (apenas grupos válidos), conforme tabela do trabalho."""
    n = len(df_anon)
    sizes = group_sizes(df_anon, qi_cols)
    valid_mask = sizes >= k
    dm_valid = float((sizes[valid_mask] ** 2).sum())
    return dm_valid / (n * n) if n > 0 else 0.0
