"""
backtest.py — оценка качества модели на исторических данных.

Идея: прогоняем модель по УЖЕ известным парам кварталов (Q_t -> Q_{t+1}) и
сравниваем прогноз с фактом из отчёта Q_{t+1}. Сценарий (scenario.json) не нужен:
фактические решения следующего квартала уже лежат в его отчёте, а модель всё
равно реагирует лишь на цену (по ячейкам) и рекламу (одно число на компанию) —
их берём из файла напрямую, без потерь. Прочие решения (НИОКР, производство,
per-cell/имидж-реклама) в FEATURES не входят -> их влияние = ошибка модели,
которую бэктест и измеряет.

Ошибку раскладываем на две стадии (ломаются по-разному):
  • стадия 1 (доля): предсказанная доля vs фактическая — чистый тест логита;
  • стадия 2 (объём): sold_t/share_t × сезонность vs sold_{t+1}/share_{t+1}.
Сквозной спрос = доля × объём, против бейзлайнов (персистенция, сезонный наив).

Два режима подачи входов:
  • realistic — конкуренты заморожены на Q_t, меняем только свои цену+рекламу
    (ровно то, что делает `forecast`); включает допущение «конкуренты не двигаются»;
  • oracle — подаём фактическое рыночное состояние Q_{t+1} целиком; изолирует
    качество функции доли (верхняя граница точности).
Разрыв между режимами = цена незнания ходов конкурентов.

Оговорка: стадия-1 обучается in-sample (модель доли фитится на --train, по
умолчанию = все --reports). Это тест «фит + перенос вперёд», не строгий OOS;
для честного OOS передай в --train только чужие группы/кварталы.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Any
import numpy as np
import pandas as pd
from .parser import parse_report, CHANNELS
from .model import load_panel, ShareModel, fit_seasonality, cell_volume


def _fin(x: Any) -> float | None:
    """Приводит к float, если это конечное число, иначе None."""
    try:
        v = float(x)
    except TypeError, ValueError:
        return None
    return v if np.isfinite(v) else None


def _mae(pred: list[float | None], act: list[float | None]) -> float | None:
    """Средняя абсолютная ошибка по парам, где оба значения определены."""
    pa = [(p, a) for p, a in zip(pred, act) if p is not None and a is not None]
    if not pa:
        return None
    return float(np.mean([abs(p - a) for p, a in pa]))


def _mape(pred: list[float | None], act: list[float | None]) -> float | None:
    """MAPE (%) по парам, где факт определён и не ноль."""
    pa = [
        (p, a)
        for p, a in zip(pred, act)
        if p is not None and a is not None and a != 0
    ]
    if not pa:
        return None
    return float(np.mean([abs(p - a) / abs(a) for p, a in pa]) * 100)


# 9 ячеек в устойчивом порядке (канал × продукт) для по-ячеечной сводки
CELLS = [(ch, p) for ch in CHANNELS for p in (1, 2, 3)]


def per_cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Метрики backtest, разложенные по 9 ячейкам (канал × продукт).
    df — детализация из backtest(). Каждая строка: доля-MAE (п.п., своя
    компания, real/oracle), объём-MAPE и сквозной спрос-MAPE (%) против
    бейзлайнов. Метрика «все 8 компаний» здесь не считается (в детализации
    только своя ячейка)."""

    def col(g: pd.DataFrame, name: str) -> list[float | None]:
        return [_fin(x) for x in g[name].tolist()]

    rows: list[dict[str, Any]] = []
    for ch, p in CELLS:
        g = df[(df["канал"] == ch) & (df["продукт"] == p)]
        if g.empty:
            continue
        rows.append(
            {
                "канал": ch,
                "продукт": p,
                "n": len(g),
                "доля_MAE_real": _mae(
                    col(g, "доля_real"), col(g, "доля_факт")
                ),
                "доля_MAE_oracle": _mae(
                    col(g, "доля_oracle"), col(g, "доля_факт")
                ),
                "объём_MAPE": _mape(
                    col(g, "объём_прогноз"), col(g, "объём_факт")
                ),
                "спрос_MAPE_real": _mape(
                    col(g, "спрос_real"), col(g, "спрос_факт")
                ),
                "спрос_MAPE_oracle": _mape(
                    col(g, "спрос_oracle"), col(g, "спрос_факт")
                ),
                "спрос_MAPE_persist": _mape(
                    col(g, "спрос_persist"), col(g, "спрос_факт")
                ),
                "спрос_MAPE_seasnaive": _mape(
                    col(g, "спрос_seasnaive"), col(g, "спрос_факт")
                ),
            }
        )
    return pd.DataFrame(rows)


def _series(reports: list[str]) -> list[tuple[str, dict[str, Any], str]]:
    """Пары смежных кварталов (cur_file, meta_cur, next_file) внутри одной
    серии (одинаковые группа+компания), отсортированные по времени."""
    groups: dict[tuple[int, int], list[tuple[int, str, dict[str, Any]]]] = (
        defaultdict(list)
    )
    for f in reports:
        m = parse_report(f)[0]
        t = int(m["year"]) * 4 + int(m["quarter"])
        groups[(int(m["group"]), int(m["company"]))].append((t, f, m))
    pairs: list[tuple[str, dict[str, Any], str]] = []
    for items in groups.values():
        items.sort(key=lambda x: x[0])
        for (t1, f1, m1), (t2, f2, _) in zip(items, items[1:]):
            if t2 == t1 + 1:  # только смежные кварталы
                pairs.append((f1, m1, f2))
    return pairs


def _own_share(
    model: ShareModel,
    cell: pd.DataFrame,
    company: int,
    price: float | None = None,
    adspend: float | None = None,
) -> float | None:
    """Предсказанная доля (%) своей компании в ячейке. price/adspend — опц.
    переопределение своих рычагов (остальные компании как в cell)."""
    c = cell.dropna(subset=["price"]).copy()
    mask = c["company"] == company
    if not mask.any():
        return None
    c["price"] = c["price"].astype(float)
    c["adspend"] = c["adspend"].astype(float)
    if price is not None:
        c.loc[mask, "price"] = float(price)
    if adspend is not None:
        c.loc[mask, "adspend"] = float(adspend)
    shares = model.predict_shares(c)
    own = shares[mask.to_numpy()]
    if len(own) == 0:
        return None
    return _fin(own[0])


def _own_val(df: pd.DataFrame, ch: str, p: int, col: str) -> float | None:
    row = df[(df["channel"] == ch) & (df["product"] == p)]
    return _fin(row[col].iloc[0]) if len(row) else None


def _panel_own_adspend(cell: pd.DataFrame, company: int) -> float | None:
    row = cell[cell["company"] == company]
    return _fin(row["adspend"].iloc[0]) if len(row) else None


def backtest(
    reports: list[str],
    train: list[str] | None = None,
    history: list[str] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Прогон модели по всем смежным парам кварталов в reports.
    Возвращает (сводка, детализация по ячейкам)."""
    pairs = _series(reports)
    if not pairs:
        raise ValueError(
            "не нашлось ни одной пары смежных кварталов "
            "(нужны отчёты одной серии за соседние кварталы)"
        )
    model = ShareModel().fit(load_panel(train or reports))
    seas = fit_seasonality(history) if history else None

    detail: list[dict[str, Any]] = []
    share_err_all: list[float] = []  # |доля_oracle − факт| по всем 8 компаниям

    for cur, m, nxt in pairs:
        company = int(m["company"])
        q_now, group = int(m["quarter"]), int(m["group"])
        q_next = int(parse_report(nxt)[0]["quarter"])
        seas_ratio = (seas[q_next] / seas[q_now]) if seas else 1.0

        panel_cur = load_panel([cur])
        panel_next = load_panel([nxt])
        own_cur = parse_report(cur)[1]
        own_next = parse_report(nxt)[1]

        for ch in CHANNELS:
            for p in (1, 2, 3):
                cur_cell = panel_cur[
                    (panel_cur["channel"] == ch) & (panel_cur["product"] == p)
                ]
                next_cell = panel_next[
                    (panel_next["channel"] == ch)
                    & (panel_next["product"] == p)
                ]

                # стадия 1, oracle по всем 8 компаниям (фит логита на факте Q_next)
                nc = next_cell.dropna(subset=["price"])
                if len(nc):
                    pred_all = model.predict_shares(nc)
                    for sp, sa in zip(pred_all, nc["share"].tolist()):
                        fp, fa = _fin(sp), _fin(sa)
                        if fp is not None and fa is not None:
                            share_err_all.append(abs(fp - fa))

                # факты Q_next по своей компании
                demand_next = _own_val(own_next, ch, p, "demand")
                share_next = _own_val(own_next, ch, p, "share_own")
                sold_next = _own_val(own_next, ch, p, "sold")
                price_next = _own_val(own_next, ch, p, "price")
                adspend_next = _panel_own_adspend(next_cell, company)

                # состояние Q_now по своей компании
                demand_now = _own_val(own_cur, ch, p, "demand")
                share_now = _own_val(own_cur, ch, p, "share_own")
                sold_now = _own_val(own_cur, ch, p, "sold")

                # стадия 2: объём (прогноз из Q_now × сезонность vs факт Q_next)
                vol_pred = None
                if sold_now is not None and share_now is not None:
                    v = cell_volume(sold_now, share_now)
                    vol_pred = v * seas_ratio if v is not None else None
                vol_act = (
                    cell_volume(sold_next, share_next)
                    if sold_next is not None and share_next is not None
                    else None
                )

                # доля (стадия 1) — оба режима, по своей компании
                sh_real = _own_share(
                    model, cur_cell, company, price_next, adspend_next
                )
                sh_orc = _own_share(model, next_cell, company)

                # сквозной спрос = доля × объём
                d_real = (
                    sh_real / 100 * vol_pred
                    if sh_real is not None and vol_pred is not None
                    else None
                )
                d_orc = (
                    sh_orc / 100 * vol_pred
                    if sh_orc is not None and vol_pred is not None
                    else None
                )
                # бейзлайны
                d_persist = demand_now
                d_seasnaive = (
                    demand_now * seas_ratio if demand_now is not None else None
                )

                detail.append(
                    {
                        "группа": group,
                        "компания": company,
                        "Q_now": q_now,
                        "Q_next": q_next,
                        "канал": ch,
                        "продукт": p,
                        "спрос_факт": demand_next,
                        "спрос_real": None
                        if d_real is None
                        else round(d_real),
                        "спрос_oracle": None
                        if d_orc is None
                        else round(d_orc),
                        "спрос_persist": d_persist,
                        "спрос_seasnaive": None
                        if d_seasnaive is None
                        else round(d_seasnaive),
                        "доля_факт": share_next,
                        "доля_real": sh_real,
                        "доля_oracle": sh_orc,
                        "объём_факт": None
                        if vol_act is None
                        else round(vol_act),
                        "объём_прогноз": None
                        if vol_pred is None
                        else round(vol_pred),
                    }
                )

    df = pd.DataFrame(detail)

    def col(name: str) -> list[float | None]:
        return [_fin(x) for x in df[name].tolist()]

    summary: dict[str, Any] = {
        "n_переходов": len(pairs),
        "n_ячеек": len(df),
        "группы": sorted(df["группа"].unique().tolist()),
        "компании": sorted(df["компания"].unique().tolist()),
        "сезонность": seas is not None,
        # стадия 1 — доля, MAE в процентных пунктах
        "доля_MAE_своя_real": _mae(col("доля_real"), col("доля_факт")),
        "доля_MAE_своя_oracle": _mae(col("доля_oracle"), col("доля_факт")),
        "доля_MAE_все_oracle": (
            float(np.mean(share_err_all)) if share_err_all else None
        ),
        # стадия 2 — объём, MAPE
        "объём_MAPE": _mape(col("объём_прогноз"), col("объём_факт")),
        # сквозной спрос, MAPE
        "спрос_MAPE_real": _mape(col("спрос_real"), col("спрос_факт")),
        "спрос_MAPE_oracle": _mape(col("спрос_oracle"), col("спрос_факт")),
        "спрос_MAPE_persist": _mape(col("спрос_persist"), col("спрос_факт")),
        "спрос_MAPE_seasnaive": _mape(
            col("спрос_seasnaive"), col("спрос_факт")
        ),
    }
    return summary, df
