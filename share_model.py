"""
gmc_share_model.py — вариант B: двухшаговая модель «доля рынка → спрос».

Стадия 1 (доля): агрегированный логит (MNL) через инверсию Берри.
  Доли по ячейке суммируются < 100% -> есть «внешняя опция» s0 = 1 - Σ sᵢ.
  Оценка обычным OLS:   ln(sᵢ / s0) = β·Xᵢ + FE(ячейка) + FE(группа) + ε
  Прогноз:   Aᵢ = exp(β·Xᵢ);  sᵢ = Aᵢ / (1 + Σ Aⱼ).
  Так корректно моделируется замещение: поднял свою цену -> доля утекает
  к конкурентам и во внешнюю опцию. Это и есть движок «крутить цены».

Стадия 2 (объём): implied-объём ячейки = own_sold / (own_share/100).
  Прогноз спроса ≈ прогноз_доли × объём.

Признаки берём по всем 8 компаниям из листа 'W' (схема gmc_parser_flat).
Групповые эффекты — фиксированные (дамми); с ростом числа групп -> random effects.

Зависимости: pandas, numpy, scikit-learn.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from parser_flat import SCHEMA as S, CH, _num, _stars

__all__ = [
    "load_panel",
    "ShareModel",
    "fit_seasonality",
    "cell_volume",
    "counterfactual",
    "predict_demand",
    "CH",
]

FEATURES = ["log_price", "log_adspend", "rating"]


# ---------- извлечение панели 8 компаний ----------
def load_panel(paths: list[str]) -> pd.DataFrame:
    """Полная панель: (файл, группа, кв, компания, продукт, канал) -> price/share/adspend/rating."""
    rows = []
    for path in paths:
        W = [
            _num(v)
            for v in pd.read_excel(path, sheet_name="W", header=None)
            .iloc[:, 0]
            .tolist()
        ]
        group = int(W[S["group"]]) if W[S["group"]] is not None else 0
        year, q = int(W[S["year"]]), int(W[S["quarter"]])
        for co in range(1, 9):
            adspend = W[S["adspend_base"] + 7 * (co - 1)]
            for p in (1, 2, 3):
                rating = _stars(W[S["rating_base"] + 7 * (co - 1) + (p - 1)])
                for ch, ci in CH.items():
                    off = 3 * (p - 1) + ci
                    rows.append(
                        {
                            "file": path,
                            "group": group,
                            "year": year,
                            "quarter": q,
                            "t": year * 4 + (q - 1),
                            "company": co,
                            "product": p,
                            "channel": ch,
                            "cell": f"{ch}{p}",
                            "gq": f"{group}_{year}Q{q}",
                            "price": W[S["price_base"] + 20 * (co - 1) + off],
                            "share": W[S["share_base"] + 10 * (co - 1) + off],
                            "adspend": adspend,
                            "rating": rating,
                        }
                    )
    df = pd.DataFrame(rows)
    # внешняя опция: доля непокрытого рынка в ячейке (в процентах)
    tot = df.groupby(["gq", "cell"])["share"].transform("sum")
    df["share_out"] = 100.0 - tot
    return df


# ---------- Стадия 1: логит-модель доли ----------
class ShareModel:
    def _design(self, d: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=d.index)
        X["log_price"] = np.log(d["price"].clip(lower=1))
        X["log_adspend"] = np.log(d["adspend"].fillna(0).clip(lower=0) + 1)
        X["rating"] = d["rating"].fillna(self._rating_mean)
        for c in self._cellcols:
            X[c] = (d["cell"] == c.replace("cell_", "")).astype(float)
        for g in self._grpcols:
            X[g] = (d["group"].astype(str) == g.replace("g_", "")).astype(
                float
            )
        return X[self.cols]

    def fit(self, df: pd.DataFrame) -> ShareModel:
        d = df[(df["share"] > 0) & (df["share_out"] > 0)].copy()
        self._rating_mean = d["rating"].mean()
        # фиксированные эффекты (drop_first -> базовая категория в интерсепте)
        self._cellcols = [f"cell_{c}" for c in sorted(d["cell"].unique())[1:]]
        self._grpcols = [
            f"g_{g}" for g in sorted(d["group"].astype(str).unique())[1:]
        ]
        self.cols = FEATURES + self._cellcols + self._grpcols
        X = self._design(d)
        y = np.log(d["share"] / 100) - np.log(
            d["share_out"] / 100
        )  # инверсия Берри
        self.reg = LinearRegression().fit(X, y)
        self.coef_ = dict(zip(self.cols, self.reg.coef_))
        self.r2 = self.reg.score(X, y)
        self.n = len(d)
        return self

    def attraction(self, d: pd.DataFrame) -> np.ndarray:
        """Aᵢ = exp(β·Xᵢ) относительно внешней опции."""
        return np.exp(self.reg.predict(self._design(d)))  # type: ignore[no-any-return]

    def predict_shares(self, cell_df: pd.DataFrame) -> np.ndarray:
        """Доли (в %) для одной ячейки одной группы-квартала (набор компаний)."""
        A = self.attraction(cell_df)
        return 100.0 * A / (1.0 + A.sum())  # type: ignore[no-any-return]

    def price_elasticity(self, share_pct: float) -> float:
        """Собственная эластичность доли по цене в MNL: β_price·(1 − sᵢ)."""
        coef = self.coef_["log_price"]
        return float(coef) * (1 - share_pct / 100)


# ---------- Стадия 2: объём рынка ----------
def fit_seasonality(hst_paths: list[str]) -> dict[int, float]:
    """
    Сезонные факторы объёма рынка по history-файлам группы 0 (компании-клоны с
    фикс. решениями -> вариация спроса = чистая сезонность). Оценка с отделением
    тренда:  log(demand) ~ FE(ячейка) + t + dummies(квартал).
    Возвращает {квартал: множитель}, нормированный к среднему геометрическому 1.
    """
    import math
    from parser_flat import parse_report_flat

    d = pd.concat(
        [parse_report_flat(f)[1] for f in hst_paths], ignore_index=True
    )
    d["cell"] = d["channel"] + d["product"].astype(str)
    d["t"] = d["year"] * 4 + (d["quarter"] - 1)
    d["t"] -= d["t"].min()
    d = d[d["demand"] > 0].copy()
    y = np.log(d["demand"])
    cell = pd.get_dummies(d["cell"], drop_first=True).astype(float)
    qd = pd.get_dummies(d["quarter"], prefix="q", drop_first=True).astype(
        float
    )
    X = pd.concat(
        [
            d[["t"]].reset_index(drop=True),
            cell.reset_index(drop=True),
            qd.reset_index(drop=True),
        ],
        axis=1,
    )
    coef = dict(zip(X.columns, LinearRegression().fit(X, y).coef_))
    gammas = {q: coef.get(f"q_{q}", 0.0) for q in (1, 2, 3, 4)}
    gm = math.exp(sum(gammas.values()) / 4)
    return {q: math.exp(gammas[q]) / gm for q in (1, 2, 3, 4)}


def cell_volume(own_sold: float, own_share_pct: float) -> float | None:
    """implied-объём ячейки из собственных продаж и доли."""
    if own_share_pct and own_share_pct > 0:
        return own_sold / (own_share_pct / 100)
    return None


def counterfactual(
    model: ShareModel, cell_df: pd.DataFrame, company: int, price_mult: float
) -> pd.DataFrame:
    """Что будет с долями, если компания изменит свою цену в price_mult раз."""
    base = model.predict_shares(cell_df)
    cf = cell_df.copy()
    cf["price"] = cf["price"].astype(float)
    cf.loc[cf["company"] == company, "price"] *= price_mult
    new = model.predict_shares(cf)
    out = cell_df[["company", "price", "share"]].copy()
    out["share_pred"] = base.round(2)
    out["share_cf"] = new.round(2)
    out["price_cf"] = cf["price"].values
    return out


def predict_demand(
    model: ShareModel,
    cell_df: pd.DataFrame,
    company: int,
    own_sold: float,
    own_share_now_pct: float,
    price_mult: float = 1.0,
    seasonality: dict[int, float] | None = None,
    quarter_now: int | None = None,
    quarter_next: int | None = None,
) -> dict[str, float | None]:
    """
    Прогноз собственного спроса в ячейке под сценарием.
    Объём калибруем по текущим продажам/доле; если задана сезонность (из группы 0),
    масштабируем объём на след. квартал множителем seas[next]/seas[now].
    Спрос ≈ прогноз_доли × объём.
    """
    vol = cell_volume(own_sold, own_share_now_pct)
    seas_ratio = 1.0
    if seasonality and quarter_now and quarter_next:
        seas_ratio = seasonality[quarter_next] / seasonality[quarter_now]
    if vol is not None:
        vol *= seas_ratio
    cf = cell_df.copy()
    cf["price"] = cf["price"].astype(float)
    if price_mult != 1.0:
        cf.loc[cf["company"] == company, "price"] *= price_mult
    shares = model.predict_shares(cf)
    own_share_pred = float(shares[cf["company"].values == company][0])
    return {
        "own_share_pred_pct": round(own_share_pred, 2),
        "seas_ratio": round(seas_ratio, 3),
        "cell_volume": None if vol is None else round(vol, 0),
        "demand_pred": None
        if vol is None
        else round(own_share_pred / 100 * vol, 0),
    }


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    hst = [p for p in args if "Hst" in p]  # history группы 0 -> сезонность
    panel_files = [p for p in args if p not in hst]
    panel = load_panel(panel_files)
    m = ShareModel().fit(panel)
    print(f"Стадия 1 (логит доли): n={m.n}  R²={m.r2:.3f}")
    print("Коэффициенты (полезность логита):")
    for k in FEATURES:
        print(f"   {k:12} = {m.coef_[k]:+.3f}")
    print(
        "Групповые эффекты:",
        {k: round(v, 3) for k, v in m.coef_.items() if k.startswith("g_")},
    )
    print(
        f"Собств. эластичность доли по цене при s=8%: {m.price_elasticity(8):.2f}"
    )

    seas = fit_seasonality(hst) if hst else None
    if seas:
        print(
            "\nСезонные факторы объёма (группа 0):",
            {q: round(v, 3) for q, v in seas.items()},
        )

    # демо-контрфактик: первая ячейка последнего файла, +10% к цене компании 1
    last = panel[panel["gq"] == panel["gq"].iloc[-1]]
    cell = last[last["cell"] == last["cell"].iloc[0]]
    print(
        f"\nКонтрфактик (ячейка {cell['cell'].iloc[0]}, группа {cell['group'].iloc[0]}): "
        f"компания 1 поднимает цену на 10%"
    )
    print(
        counterfactual(m, cell, company=1, price_mult=1.10).to_string(
            index=False
        )
    )

    # стадия 2 со сезонностью: спрос компании 1, прогноз с текущего кв. на следующий
    from parser_flat import parse_report_flat

    meta, own = parse_report_flat(panel_files[-1])
    qn = meta["quarter"]
    qnext = qn % 4 + 1
    r = own[
        (own["channel"] == cell["channel"].iloc[0])
        & (own["product"] == int(cell["product"].iloc[0]))
    ]
    sold = float(r["sold"].iloc[0]) if len(r) else 500.0
    sh_now = float(cell[cell["company"] == 1]["share"].iloc[0])
    flat = predict_demand(m, cell, 1, sold, sh_now, 1.00)
    seasoned = predict_demand(
        m,
        cell,
        1,
        sold,
        sh_now,
        1.00,
        seasonality=seas,
        quarter_now=qn,
        quarter_next=qnext,
    )
    print(f"\nСтадия 2, спрос компании 1 (Q{qn}->Q{qnext}):")
    print(f"   без сезонности: {flat}")
    print(f"   с сезонностью : {seasoned}")
