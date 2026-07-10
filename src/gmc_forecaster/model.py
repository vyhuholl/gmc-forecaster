"""
gmc_model.py — прозрачная модель прогноза спроса t+1.

Модель: log(demand_next) = a + b·log(lag1) + c·d_price_rel
                              + d·log(ad_product+1) + e·cum_major + f·minor_since
Логика: b — авторегрессия (инерция/рост), c — эластичность по относительной цене,
d — отдача рекламы, e/f — эффект новых разработок. Всё интерпретируемо и
защищаемо на интервью. На малых данных линейная модель предпочтительнее бустинга.

Интерфейс намеренно как у sklearn: fit(df) / predict(df).
"""

from __future__ import annotations
from typing import cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

MODEL_FEATURES = [
    "log_lag1",
    "d_price_rel",
    "log_ad",
    "cum_major",
    "minor_since",
]


def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["log_lag1"] = np.log(df["lag1"].clip(lower=1))
    X["d_price_rel"] = df["d_price_rel"].fillna(0.0)
    X["log_ad"] = np.log(df["ad_product"].fillna(0.0) + 1)
    X["cum_major"] = df["cum_major"].fillna(0)
    X["minor_since"] = df["minor_since"].fillna(0)
    return X


class DemandModel:
    def __init__(self) -> None:
        self.reg = LinearRegression()

    def fit(self, df: pd.DataFrame) -> DemandModel:
        X = _design(df)
        y = np.log(df["demand_next"].clip(lower=1))
        self.reg.fit(X, y)
        self.coef_ = dict(zip(MODEL_FEATURES, self.reg.coef_))
        self.intercept_ = self.reg.intercept_
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.reg.predict(_design(df)))  # type: ignore[no-any-return]


def _mape(y: np.ndarray, p: np.ndarray) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean(np.abs((y - p) / y)) * 100)


def backtest(
    df: pd.DataFrame, holdout_quarter: int
) -> tuple[dict[str, object], pd.DataFrame]:
    """Обучение на кварталах < holdout, прогноз на holdout. Сравнение с наивными."""
    tr = df[df["quarter"] < holdout_quarter]
    te = df[df["quarter"] == holdout_quarter]
    m = DemandModel().fit(tr)
    pred = m.predict(te)
    y = cast(np.ndarray, te["demand_next"].values)
    naive_persist = cast(np.ndarray, te["lag1"].values)  # спрос не меняется
    g = float(
        (tr["demand_next"] / tr["lag1"]).mean()
    )  # средний рост на обучении
    naive_growth = cast(np.ndarray, te["lag1"].values) * g
    return {
        "n_train": len(tr),
        "n_test": len(te),
        "MAPE_model": round(_mape(y, pred), 1),
        "MAPE_persist": round(_mape(y, naive_persist), 1),
        "MAPE_growth": round(_mape(y, naive_growth), 1),
        "coef": {k: round(v, 3) for k, v in m.coef_.items()},
    }, te.assign(pred=pred.round(0), truth=y)
