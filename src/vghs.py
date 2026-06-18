"""Hierarquias de Generalização (Value Generalization Hierarchies)
para os QIs do Adult Dataset, conforme Tabela 2 do trabalho."""
import pandas as pd

VGH_LEVELS = {
    "age": 4,
    "workclass": 3,
    "education": 3,
    "marital_status": 3,
    "occupation": 3,
    "race": 2,
    "sex": 2,
    "native_country": 3,
}

QI_ORDER = [
    "age", "workclass", "education", "marital_status",
    "occupation", "race", "sex", "native_country",
]

WORKCLASS_MAP = {
    "Federal-gov": "Publico", "Local-gov": "Publico", "State-gov": "Publico",
    "Private": "Privado",
    "Self-emp-inc": "Autonomo", "Self-emp-not-inc": "Autonomo",
    "Without-pay": "Outro", "Never-worked": "Outro",
}

EDUCATION_MAP = {
    "Preschool": "Basico", "1st-4th": "Basico", "5th-6th": "Basico",
    "7th-8th": "Basico", "9th": "Basico", "10th": "Basico",
    "11th": "Basico", "12th": "Basico",
    "HS-grad": "Medio", "Some-college": "Medio",
    "Assoc-acdm": "Medio", "Assoc-voc": "Medio", "Prof-school": "Medio",
    "Bachelors": "Superior", "Masters": "Superior", "Doctorate": "Superior",
}

MARITAL_MAP = {
    "Married-civ-spouse": "Casado", "Married-AF-spouse": "Casado",
    "Never-married": "Solteiro",
    "Divorced": "Outro", "Separated": "Outro",
    "Widowed": "Outro", "Married-spouse-absent": "Outro",
}

OCCUPATION_MAP = {
    "Exec-managerial": "White-Collar", "Prof-specialty": "White-Collar",
    "Adm-clerical": "White-Collar", "Sales": "White-Collar",
    "Tech-support": "White-Collar",
    "Craft-repair": "Blue-Collar", "Machine-op-inspct": "Blue-Collar",
    "Transport-moving": "Blue-Collar", "Handlers-cleaners": "Blue-Collar",
    "Farming-fishing": "Blue-Collar",
    "Other-service": "Servicos", "Priv-house-serv": "Servicos",
    "Protective-serv": "Servicos", "Armed-Forces": "Servicos",
}

CONTINENT_MAP = {
    "United-States": "America-Norte", "Canada": "America-Norte",
    "Outlying-US(Guam-USVI-etc)": "America-Norte",
    "Mexico": "America-Central", "Puerto-Rico": "America-Central",
    "Cuba": "America-Central", "Jamaica": "America-Central",
    "Dominican-Republic": "America-Central", "Haiti": "America-Central",
    "Honduras": "America-Central", "Guatemala": "America-Central",
    "Nicaragua": "America-Central", "El-Salvador": "America-Central",
    "Trinadad&Tobago": "America-Central",
    "Columbia": "America-Sul", "Ecuador": "America-Sul", "Peru": "America-Sul",
    "England": "Europa", "Germany": "Europa", "Italy": "Europa",
    "Poland": "Europa", "Portugal": "Europa", "France": "Europa",
    "Ireland": "Europa", "Greece": "Europa", "Yugoslavia": "Europa",
    "Scotland": "Europa", "Hungary": "Europa", "Holand-Netherlands": "Europa",
    "India": "Asia", "China": "Asia", "Japan": "Asia", "Vietnam": "Asia",
    "Philippines": "Asia", "Cambodia": "Asia", "Thailand": "Asia",
    "Laos": "Asia", "Taiwan": "Asia", "Hong": "Asia", "Iran": "Asia",
    "South": "Asia",
}


def _gen_age(series: pd.Series, level: int) -> pd.Series:
    if level == 0:
        return series.astype(str)
    if level == 1:
        bins = list(range(0, 105, 5))
        labels = [f"[{b},{b+5})" for b in bins[:-1]]
        return pd.cut(series, bins=bins, right=False, labels=labels).astype(str)
    if level == 2:
        bins = list(range(0, 110, 10))
        labels = [f"[{b},{b+10})" for b in bins[:-1]]
        return pd.cut(series, bins=bins, right=False, labels=labels).astype(str)
    return pd.Series(["*"] * len(series), index=series.index)


def _gen_category(series: pd.Series, level: int, mapping: dict, max_level: int) -> pd.Series:
    if level == 0:
        return series.astype(str)
    if level >= max_level:
        return pd.Series(["*"] * len(series), index=series.index)
    return series.map(lambda v: mapping.get(v, "Outro"))


def _gen_binary(series: pd.Series, level: int) -> pd.Series:
    if level == 0:
        return series.astype(str)
    return pd.Series(["*"] * len(series), index=series.index)


def apply_generalization(df: pd.DataFrame, chromosome: list[int]) -> pd.DataFrame:
    """Aplica a generalização codificada pelo cromossomo nas colunas QI.
    `chromosome[i]` é o nível VGH para QI_ORDER[i]."""
    out = df.copy()
    out["age"] = _gen_age(df["age"], chromosome[0])
    out["workclass"] = _gen_category(df["workclass"], chromosome[1], WORKCLASS_MAP, 2)
    out["education"] = _gen_category(df["education"], chromosome[2], EDUCATION_MAP, 2)
    out["marital_status"] = _gen_category(df["marital_status"], chromosome[3], MARITAL_MAP, 2)
    out["occupation"] = _gen_category(df["occupation"], chromosome[4], OCCUPATION_MAP, 2)
    out["race"] = _gen_binary(df["race"], chromosome[5])
    out["sex"] = _gen_binary(df["sex"], chromosome[6])
    out["native_country"] = _gen_category(df["native_country"], chromosome[7], CONTINENT_MAP, 2)
    return out


def total_search_space() -> int:
    total = 1
    for v in VGH_LEVELS.values():
        total *= v
    return total
