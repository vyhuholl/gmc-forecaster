"""
вариант B: двухшаговая модель «доля рынка → спрос».

Стадия 1 (доля): агрегированный логит (MNL) через инверсию Берри.
  Доли по ячейке суммируются < 100% -> есть «внешняя опция» s0 = 1 - Σ sᵢ.
  Модель:  ln(sᵢ / s0) = β·Xᵢ + FE(ячейка) + FE(группа)
                         + FE(фирма, центр. внутри группы)
                         + (β_price + u_g)·log_price + ε
  • Firm-эффект ловит устойчивую фирменную гетерогенность (бренд/дистрибуция),
    которую цена/рейтинг не объясняют; это уровень, не наклон. Центрирование
    внутри группы -> незнакомая фирма откатывается на групповой эффект.
  • Random slope u_g: наклон цены гетерогенен по группам (эмпирически −2.5…−4.9).
    u_g усаживаются к global штрафом (ridge), λ выбирается leave-one-quarter-out
    CV -> хорошо наблюдаемая группа берёт свой наклон, разреженная/новая (мало
    кварталов -> не идентифицируется) откатывается к global. Это partial pooling.
  Оценка — штрафованный МНК (штраф только на u_g). Прогноз: Aᵢ = exp(β·Xᵢ);
  sᵢ = Aᵢ / (1 + Σ Aⱼ). Так корректно моделируется замещение: поднял свою цену
  -> доля утекает к конкурентам и во внешнюю опцию. Это движок «крутить цены».

Стадия 2 (объём): implied-объём ячейки = own_sold / (own_share/100).
  Прогноз спроса ≈ прогноз_доли × объём.

Признаки берём по всем 8 компаниям из листа 'W'. Групповые эффекты и наклоны —
фикс. дамми + усадка (partial pooling); с ростом числа групп -> полноценные RE.

Зависимости: pandas, numpy, scikit-learn.
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .parser import SCHEMA as S, CH, _num, _stars

__all__ = [
    "load_panel",
    "ShareModel",
    "fit_seasonality",
    "cell_volume",
    "counterfactual",
    "predict_demand",
    "damp_lever",
    "LEVER_K",
    "CH",
]

FEATURES = ["log_price", "log_adspend", "rating"]

# ограничение причинного рычага заякоренного прогноза: защита от взрыва доли
# при вырожденном фите (напр. обучение на одной группе с одним кварталом долей)
LEVER_CLIP = (1 / 3, 3.0)

# дефолтный демпфер причинного рычага заякоренного прогноза (см. damp_lever).
# 0.5: полная эластичность логита переоценена и OOS не обобщается (§14.5/§15) —
# на бэктесте 2026 k=0.5 робастнее по группам (улучшает gr.11, смягчает провал
# gr.13/gr.2), чем полный рычаг k=1.0, ценой ~0.5 п.п. агрегата.
LEVER_K = 0.5


def damp_lever(ratio: float, k: float) -> float:
    """Демпфированный (к 1.0 на долю 1−k) и ограниченный причинный рычаг.
    ratio — модельное отношение доли сцен/база; k — сила рычага (1.0 полный,
    0.0 нейтральный = сезонный наив). Клип LEVER_CLIP отсекает нефизичные
    множители спроса от вырожденных фитов модели доли."""
    lev = 1.0 + (ratio - 1.0) * k
    return min(max(lev, LEVER_CLIP[0]), LEVER_CLIP[1])


# ---------- извлечение панели 8 компаний ----------
def load_panel(paths: list[str]) -> pd.DataFrame:
    """Полная панель: (файл, группа, кв, компания, продукт, канал) -> price/share/adspend/rating."""
    rows = []
    for path in paths:
        W = [
            _num(v)
            for v in pd.read_excel(
                path, sheet_name="W", header=None, engine="calamine"
            )
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
def _firm(d: pd.DataFrame) -> pd.Series:
    """Идентификатор фирмы = группа_компания (напр. '11_5')."""
    return d["group"].astype(str) + "_" + d["company"].astype(str)


# человекочитаемые метки/блоки коэффициентов (для coef_summary и вывода)
_COEF_LABELS = {
    "const": "Константа",
    "log_price": "log(цена)",
    "log_adspend": "log(реклама+1)",
    "rating": "рейтинг (звёзды)",
}
_COEF_PREFIX = {
    "cell_": ("ячейка ", "ячейка"),
    "g_": ("группа ", "группа"),
    "firm_": ("фирма ", "фирма"),
    "lp_g_": ("наклон Δ группы ", "наклон"),
}


def _coef_label(col: str) -> str:
    if col in _COEF_LABELS:
        return _COEF_LABELS[col]
    for pre, (lbl, _) in _COEF_PREFIX.items():
        if col.startswith(pre):
            return lbl + col[len(pre) :]
    return col


def _coef_block(col: str) -> str:
    if col == "const":
        return "const"
    if col in FEATURES:
        return "признак"
    for pre, (_, blk) in _COEF_PREFIX.items():
        if col.startswith(pre):
            return blk
    return "прочее"


class ShareModel:
    def _design(self, d: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(index=d.index)
        X["const"] = 1.0
        X["log_price"] = np.log(d["price"].clip(lower=1))
        X["log_adspend"] = np.log(d["adspend"].fillna(0).clip(lower=0) + 1)
        X["rating"] = d["rating"].fillna(self._rating_mean)
        for c in self._cellcols:
            X[c] = (d["cell"] == c.replace("cell_", "")).astype(float)
        for g in self._grpcols:
            X[g] = (d["group"].astype(str) == g.replace("g_", "")).astype(
                float
            )
        # firm-эффекты, центрированные ВНУТРИ группы (эффект-кодирование):
        # реф-фирма группы = −1 по всем её столбцам, прочие = 0/1. Незнакомая
        # фирма -> все столбцы 0 -> откат на групповой эффект (см. fit).
        firm = _firm(d)
        for col, (f, ref) in self._firm_spec.items():
            X[col] = np.where(firm == f, 1.0, np.where(firm == ref, -1.0, 0.0))
        # групповые ДЕВИАЦИИ наклона цены u_g (random slope): эффект. наклон
        # группы g = global log_price + u_g. Незнакомая группа -> нули столбцов
        # -> global (усадка задаётся штрафом на u_g, см. fit/_choose_ridge).
        grp = d["group"].astype(str)
        lp = X["log_price"]
        for g in self._slope_groups:
            X[f"lp_g_{g}"] = np.where(grp == g, lp, 0.0)
        return X[self.cols]

    @staticmethod
    def _solve(X: np.ndarray, y: np.ndarray, pen: np.ndarray) -> np.ndarray:
        """Штрафованный МНК: min ||Xβ−y||² + Σ penⱼ·βⱼ² как расширенная задача
        наименьших квадратов lstsq([X; diag(√pen)], [y; 0]). Штраф на u_g>0
        снимает коллинеарность global-наклона с групповыми девиациями (усадка)."""
        sq = np.sqrt(pen)
        Xa = np.vstack([X, np.diag(sq)])
        ya = np.concatenate([y, np.zeros(len(pen))])
        sol: np.ndarray = np.linalg.lstsq(Xa, ya, rcond=None)[0]
        return sol

    def _choose_ridge(
        self, X: np.ndarray, y: np.ndarray, base: np.ndarray, gq: np.ndarray
    ) -> float:
        """λ усадки групповых наклонов через leave-one-quarter-out CV по gq
        (обобщается ли свой наклон группы на её невиданный квартал). Мало
        групп/кварталов -> сильная усадка к global."""
        units = np.unique(gq)
        if len(self._slope_groups) < 2 or len(units) < 2:
            return 1e6
        grid = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
        best_lam, best_err = grid[-1], float("inf")
        for lam in grid:
            pen = base * lam
            err = 0.0
            for u in units:
                tr = gq != u
                beta = self._solve(X[tr], y[tr], pen)
                r = y[~tr] - X[~tr] @ beta
                err += float(r @ r)
            if err < best_err:
                best_err, best_lam = err, lam
        return best_lam

    def fit(self, df: pd.DataFrame) -> ShareModel:
        d = df[(df["share"] > 0) & (df["share_out"] > 0)].copy()
        self._rating_mean = d["rating"].mean()
        # фиксированные эффекты (drop_first -> базовая категория в const)
        self._cellcols = [f"cell_{c}" for c in sorted(d["cell"].unique())[1:]]
        self._grpcols = [
            f"g_{g}" for g in sorted(d["group"].astype(str).unique())[1:]
        ]
        # firm-девиации от среднего своей группы (эффект-кодирование): реф-фирма
        # каждой группы кодируется как −Σ остальных, поэтому незнакомая фирма
        # даёт нулевую девиацию и наследует групповой уровень.
        self._firm_spec: dict[str, tuple[str, str]] = {}
        for _, sub in d.groupby(d["group"].astype(str)):
            firms = sorted(_firm(sub).unique())
            if len(firms) < 2:  # одна фирма -> девиация не определена
                continue
            ref = firms[-1]
            for f in firms[:-1]:
                self._firm_spec[f"firm_{f}"] = (f, ref)
        # девиации наклона цены — по ВСЕМ группам (коллинеарность с global
        # снимается штрафом при усадке)
        self._slope_groups = sorted(d["group"].astype(str).unique())
        self.cols = (
            ["const"]
            + FEATURES
            + self._cellcols
            + self._grpcols
            + list(self._firm_spec)
            + [f"lp_g_{g}" for g in self._slope_groups]
        )
        X = self._design(d).to_numpy(dtype=float)
        y = (np.log(d["share"] / 100) - np.log(d["share_out"] / 100)).to_numpy(
            dtype=float
        )  # инверсия Берри
        # штраф только на девиации наклона u_g (усадка random slope)
        base = np.array(
            [1.0 if c.startswith("lp_g_") else 0.0 for c in self.cols]
        )
        self.ridge_ = self._choose_ridge(X, y, base, d["gq"].to_numpy())
        pen = base * self.ridge_
        self.beta = self._solve(X, y, pen)
        self.coef_ = dict(zip(self.cols, self.beta))
        resid = y - X @ self.beta
        tss = float(((y - y.mean()) ** 2).sum())
        self.r2 = 1.0 - float(resid @ resid) / tss if tss > 0 else 0.0
        self.n = len(d)
        self._infer(X, resid, pen)
        return self

    def _infer(
        self, X: np.ndarray, resid: np.ndarray, pen: np.ndarray
    ) -> None:
        """Стандартные ошибки/значимость коэффициентов.
        Ковариация штрафованного МНК (сэндвич): Cov = σ²·Ainv·(XᵀX)·Ainv,
        Ainv = (XᵀX + diag(pen))⁻¹ — та же матрица штрафа, что и в оценке. Для
        непенализуемых признаков (цена/реклама/рейтинг/FE) сводится к обычной
        OLS-ковариации σ²·(XᵀX)⁻¹; для наклонов u_g учитывает усадку. σ² берём
        на эффективном числе ст. свободы n − edf, edf = tr(Ainv·XᵀX) («шляпа»
        гребневой регрессии). p-value — двусторонний, нормальное приближение
        (erfc); для штрафуемых u_g значимость приблизительная."""
        XtX = X.T @ X
        Ainv = np.linalg.pinv(XtX + np.diag(pen))
        S1 = Ainv @ XtX
        self.edf_ = float(np.trace(S1))
        dof = max(self.n - self.edf_, 1.0)
        sigma2 = float(resid @ resid) / dof
        var = np.clip(np.diag(sigma2 * (S1 @ Ainv)), 0.0, None)
        se = np.sqrt(var)
        tstat = np.divide(self.beta, se, out=np.zeros_like(se), where=se > 0)
        pval = np.array([math.erfc(abs(t) / math.sqrt(2)) for t in tstat])
        self.se_ = dict(zip(self.cols, se))
        self.tstat_ = dict(zip(self.cols, tstat))
        self.pval_ = dict(zip(self.cols, pval))

    def coef_summary(self) -> pd.DataFrame:
        """Таблица коэффициентов стадии 1 со ст. ошибками и значимостью:
        столбцы col/блок/признак/коэф/ст.ош/t/p. Блок группирует признаки
        (const/признак/наклон/ячейка/группа/фирма) для читаемого вывода."""
        return pd.DataFrame(
            {
                "col": c,
                "блок": _coef_block(c),
                "признак": _coef_label(c),
                "коэф": float(self.coef_[c]),
                "ст.ош": float(self.se_[c]),
                "t": float(self.tstat_[c]),
                "p": float(self.pval_[c]),
            }
            for c in self.cols
        )

    def attraction(self, d: pd.DataFrame) -> np.ndarray:
        """Aᵢ = exp(β·Xᵢ) относительно внешней опции."""
        X = self._design(d).to_numpy(dtype=float)
        return np.exp(X @ self.beta)  # type: ignore[no-any-return]

    def predict_shares(
        self, cell_df: pd.DataFrame, attr_mult: np.ndarray | None = None
    ) -> np.ndarray:
        """Доли (в %) для одной ячейки одной группы-квартала (набор компаний).
        attr_mult — поэлементный множитель привлекательности Aᵢ (напр. эффект
        дистрибьюторов у своей компании); None -> без изменений.
        Неактивные компании (цена ≤ 0 или пусто — не предлагают продукт)
        исключаются из конкуренции и получают долю 0: иначе клип цены 0→1 дал бы
        фантомно «дешёвого» игрока, забирающего почти всю долю ячейки."""
        price = pd.to_numeric(cell_df["price"], errors="coerce").to_numpy()
        active = np.isfinite(price) & (price > 0)
        shares = np.zeros(len(cell_df))
        if not active.any():
            return shares
        A = self.attraction(cell_df[active])
        if attr_mult is not None:
            A = A * np.asarray(attr_mult)[active]
        shares[active] = 100.0 * A / (1.0 + A.sum())
        return shares

    def price_elasticity(
        self, share_pct: float, group: str | int | None = None
    ) -> float:
        """Собственная эластичность доли по цене в MNL: β_price·(1 − sᵢ).
        Наклон группы = global log_price + u_g (если группа известна), иначе
        global."""
        coef = self.coef_["log_price"]
        if group is not None:
            coef += self.coef_.get(f"lp_g_{group}", 0.0)
        return float(coef) * (1 - share_pct / 100)


# ---------- Стадия 2: объём рынка ----------
def fit_seasonality(hst_paths: list[str]) -> dict[int, float]:
    """
    Сезонные факторы объёма рынка по history-файлам группы 0 (компании-клоны с
    фикс. решениями -> вариация спроса = чистая сезонность). Оценка с отделением
    тренда:  log(demand) ~ FE(ячейка) + t + dummies(квартал).
    Возвращает {квартал: множитель}, нормированный к среднему геометрическому 1.
    """
    from .parser import parse_report

    d = pd.concat([parse_report(f)[1] for f in hst_paths], ignore_index=True)
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
