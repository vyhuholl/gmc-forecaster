"""
gmc_features.py — из истории отчётов строит матрицу признаков для прогноза
спроса следующего квартала (target=demand_next).

Признаки, которым нужна ИСТОРИЯ (поэтому считаем после склейки, по каждой
ячейке channel×product, отсортированной по времени t):
  - lag1        : спрос текущего кв. (сильнейший предиктор, авторегрессия)
  - d_price_rel : изменение относительной цены (own/среднее конкурентов)
  - ad_adstock  : имидж-реклама с геометрическим распадом (4 кв.)
  - cum_major   : накопленное число Major с начала
  - minor_since : число Minor с последнего Major
  - dist_n, dist_comm, gdp — как есть
"""

from __future__ import annotations
import pandas as pd
from .parser_flat import parse_report_flat as parse_report

IMAGE_DECAY = 0.5  # λ распада имидж-рекламы; подбирается по данным


def _cell_features(g: pd.DataFrame) -> pd.DataFrame:
    """g — одна ячейка (channel,product), отсортирована по t возрастанию."""
    g = g.sort_values("t").copy()
    g["lag1"] = g["demand"]
    g["d_price_rel"] = g["price_rel"].diff()
    # adstock имиджа: A_t = image_t + λ*A_{t-1}
    a, adstock = 0.0, []
    for x in g["ad_image"].fillna(0.0):
        a = x + IMAGE_DECAY * a
        adstock.append(a)
    g["ad_adstock"] = adstock
    # накопленные разработки
    cum_major, minor_since, cm, ms = [], [], 0, 0
    for d in g["new_dev"].fillna(""):
        if d == "Major":
            cm += 1
            ms = 0
        elif d == "Minor":
            ms += 1
        cum_major.append(cm)
        minor_since.append(ms)
    g["cum_major"] = cum_major
    g["minor_since"] = minor_since
    return g


def build_features(paths: list[str]) -> pd.DataFrame:
    """Склейка файлов -> таблица переходов с признаками; строки без t+1 отброшены."""
    frames = [parse_report(p)[1] for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["t"] = df["year"] * 4 + (df["quarter"] - 1)
    # признаки, требующие истории, — по каждой ячейке (channel, product)
    df = pd.concat(
        [_cell_features(g) for _, g in df.groupby(["channel", "product"])],
        ignore_index=True,
    )
    df["demand_next"] = df.groupby(["channel", "product"])["demand"].shift(-1)
    return df.dropna(subset=["demand_next"]).reset_index(drop=True)


FEATURES = [
    "lag1",
    "price_rel",
    "d_price_rel",
    "ad_product",
    "ad_adstock",
    "dist_n",
    "dist_comm",
    "cum_major",
    "minor_since",
]
