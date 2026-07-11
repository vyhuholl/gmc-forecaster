"""
forecast.py — прогноз спроса на следующий квартал под сценарием решений.
Это рабочий интерфейс «крутить цены/рекламу»: меняешь свои цены -> видишь спрос
по всем 9 ячейкам (3 продукта × 3 канала).

Пайплайн:
  1. Обучаем логит-модель доли на конкурентных отчётах (train).
  2. Оцениваем сезонность объёма по группе 0 (history, опционально).
  3. Берём текущее состояние рынка из current (все 8 компаний + свои продажи/доля).
  4. Применяем сценарий (новые цены / множитель рекламы) к своей компании.
  5. По каждой ячейке: доля(сценарий) × объём(с сезонной поправкой) = спрос.

Формат scenario.json:
  {
    "quarter_next": 1,
    "price":   {"EAEU1": 320, "EAEU2": 580, "INT3": 740},   // абс. цены, частично
    "adspend_mult": 1.15                                     // опц. множитель рекламы
  }
Незаданные ячейки сохраняют текущие значения.
"""

from __future__ import annotations
import pandas as pd
from .parser import parse_report
from .model import (
    load_panel,
    ShareModel,
    fit_seasonality,
    cell_volume,
    CH,
)

type Scenario = dict[str, int | float | dict[str, float]]

RU = {"EAEU": "ЕАЭС", "ASEAN": "АСЕАН", "INT": "Интернет"}


def _predict_cell(
    model: ShareModel,
    cell: pd.DataFrame,
    company: int,
    sold_now: float | None,
    share_now: float | None,
    seas_ratio: float,
    price: float | None = None,
    ad_mult: float = 1.0,
) -> tuple[float, float | None]:
    c = cell.copy()
    c["price"] = c["price"].astype(float)
    c["adspend"] = c["adspend"].astype(float)
    mask = c["company"] == company
    if price is not None:
        c.loc[mask, "price"] = float(price)
    if ad_mult != 1.0:
        c.loc[mask, "adspend"] *= ad_mult
    shares = model.predict_shares(c)
    share_pred = float(shares[c["company"].values == company][0])
    if sold_now is None or share_now is None:
        return share_pred, None
    vol = cell_volume(sold_now, share_now)
    if vol is None:
        return share_pred, None
    vol = vol * seas_ratio
    demand = round(share_pred / 100 * vol)
    return share_pred, demand


def forecast(
    current: str, train: list[str], history: list[str], scenario: Scenario
) -> pd.DataFrame:
    model = ShareModel().fit(load_panel(train))
    seas = fit_seasonality(history) if history else None

    cur_panel = load_panel([current])
    meta, own = parse_report(current)
    company = meta["company"]
    q_now = meta["quarter"]
    q_next_raw = scenario.get("quarter_next", q_now % 4 + 1)
    q_next = (
        int(q_next_raw)
        if isinstance(q_next_raw, (int, float))
        else q_now % 4 + 1
    )
    seas_ratio = (seas[q_next] / seas[q_now]) if seas else 1.0
    price_sc_raw = scenario.get("price", {})
    price_sc: dict[str, float] = (
        price_sc_raw if isinstance(price_sc_raw, dict) else {}
    )
    ad_mult_raw = scenario.get("adspend_mult", 1.0)
    ad_mult = (
        float(ad_mult_raw) if isinstance(ad_mult_raw, (int, float)) else 1.0
    )

    out = []
    for ch in CH:
        for p in (1, 2, 3):
            key = f"{ch}{p}"
            cell = cur_panel[
                (cur_panel["channel"] == ch) & (cur_panel["product"] == p)
            ]
            orow = own[(own["channel"] == ch) & (own["product"] == p)].iloc[0]
            sold_now = float(orow["sold"]) if pd.notna(orow["sold"]) else None
            share_now = (
                float(orow["share_own"])
                if pd.notna(orow["share_own"])
                else None
            )
            price_now = (
                float(orow["price"]) if pd.notna(orow["price"]) else None
            )
            new_price = price_sc.get(key, price_now)

            # базовый прогноз (текущие решения) и сценарный — разница = эффект рычага
            _, d_base = _predict_cell(
                model,
                cell,
                company,
                sold_now,
                share_now,
                seas_ratio,
                price=price_now,
                ad_mult=1.0,
            )
            sh_sc, d_sc = _predict_cell(
                model,
                cell,
                company,
                sold_now,
                share_now,
                seas_ratio,
                price=new_price,
                ad_mult=ad_mult,
            )
            delta = (
                round((d_sc / d_base - 1) * 100, 1)
                if d_base and d_sc
                else None
            )
            out.append(
                {
                    "канал": RU[ch],
                    "продукт": p,
                    "спрос_текущ": int(orow["demand"])
                    if pd.notna(orow["demand"])
                    else None,
                    "цена_текущ": price_now,
                    "цена_сцен": new_price,
                    "доля_сцен_%": round(sh_sc, 2),
                    "спрос_база": d_base,
                    "спрос_сцен": d_sc,
                    "Δ_рычаг_%": delta,
                }
            )
    df = pd.DataFrame(out)
    df.attrs["meta"] = dict(
        company=company,
        group=meta["group"],
        q_now=q_now,
        q_next=q_next,
        seas_ratio=round(seas_ratio, 3),
    )
    return df
