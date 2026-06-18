"""Carga e pré-processamento do Adult Dataset (UCI ML Repository)."""
from pathlib import Path
import pandas as pd

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

QI_COLS = [
    "age", "workclass", "education", "marital_status",
    "occupation", "race", "sex", "native_country",
]

SENSITIVE_COL = "income"
NON_QI_COLS = ["fnlwgt", "capital_gain", "capital_loss", "hours_per_week"]


def load_adult(data_dir: Path | str) -> pd.DataFrame:
    data_dir = Path(data_dir)
    df_train = pd.read_csv(
        data_dir / "adult.data",
        header=None, names=COLUMNS,
        sep=r",\s*", engine="python", na_values=["?"],
    )
    df_test = pd.read_csv(
        data_dir / "adult.test",
        header=None, names=COLUMNS,
        sep=r",\s*", engine="python", na_values=["?"],
        skiprows=1,
    )
    df_test["income"] = df_test["income"].str.rstrip(".")
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.dropna(subset=QI_COLS + [SENSITIVE_COL]).reset_index(drop=True)
    df[SENSITIVE_COL] = (df[SENSITIVE_COL].str.strip() == ">50K").astype(int)
    df["education_num"] = df["education_num"].astype(int)
    df["age"] = df["age"].astype(int)
    return df
