"""Avaliação final dos cromossomos anonimizados com RandomForest."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from .vghs import QI_ORDER, apply_generalization


EXTRA_FEATURES = ["fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]


def _encode(df: pd.DataFrame, ref_columns=None):
    extra = [c for c in EXTRA_FEATURES if c in df.columns]
    X = pd.get_dummies(df[QI_ORDER + extra].copy(), columns=QI_ORDER, drop_first=False)
    if ref_columns is not None:
        X = X.reindex(columns=ref_columns, fill_value=0)
    return X


def evaluate_final(chromosome: list[int],
                   df_train: pd.DataFrame,
                   df_test: pd.DataFrame,
                   seed: int = 42) -> dict:
    """Anonimiza treino e teste com o cromossomo e treina RandomForest."""
    df_train_anon = apply_generalization(df_train, chromosome)
    df_test_anon = apply_generalization(df_test, chromosome)

    X_train = _encode(df_train_anon)
    X_test = _encode(df_test_anon, ref_columns=X_train.columns)

    y_train = df_train_anon["income"].values
    y_test = df_test_anon["income"].values

    clf = RandomForestClassifier(n_estimators=100, max_depth=None,
                                  random_state=seed, n_jobs=-1)
    clf.fit(X_train.values, y_train)
    y_pred = clf.predict(X_test.values)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }


def evaluate_original(df_train: pd.DataFrame, df_test: pd.DataFrame,
                      seed: int = 42) -> dict:
    """Baseline sem anonimização (limite superior de utilidade)."""
    return evaluate_final([0] * len(QI_ORDER), df_train, df_test, seed=seed)
