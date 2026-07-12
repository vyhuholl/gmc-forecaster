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
    "price":   {"EAEU1": 320, "EAEU2": 580, "INT3": 740},   // абс. цены, частично
    "adspend_mult": 1.15,                                    // опц. множитель рекламы
    "dist_n":    {"EAEU": 7, "ASEAN": 6},                    // опц. число дистрибьюторов по каналам
    "dist_comm": {"EAEU": 15},                               // опц. комиссия (%) по каналам
    "dist_elasticity": {"n": 0.15, "comm": 0.10}             // опц. коэффициенты эффекта
  }
Незаданные ячейки/каналы сохраняют текущие значения. Дистрибьюторы —
пер-канальные рычаги (EAEU/ASEAN/INT, действуют на все 3 продукта канала),
наблюдаемы только у своей компании -> дают множитель СОБСТВЕННОЙ
привлекательности A_own в логите доли (конкуренты не затронуты). Квартал
прогноза берётся из текущего excel-файла (следующий за кварталом на листе 'W'),
не из сценария.
"""

from __future__ import annotations
import math
import numpy as np
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

# дефолтные коэффициенты эффекта дистрибьюторов (переопределяются в сценарии)
DIST_K_N = 0.15  # чувствительность к числу дистрибьюторов (относит. reach)
DIST_K_COMM = 0.10  # чувствительность к комиссии


def _dist_mult(
    n_cur: float | None,
    n_new: float | None,
    comm_cur: float | None,
    comm_new: float | None,
    k_n: float,
    k_comm: float,
) -> float:
    """Множитель собственной привлекательности A_own от смены дистрибьюторов
    относительно текущего решения (=1.0, если ничего не меняли).
      reach:   1 + k_n · ln((1+n_new)/(1+n_cur))
      комиссия: 1 + k_comm · (comm_new/comm_cur − 1)
    Пер-канальный рычаг: одинаков для всех 3 продуктов канала."""
    mult = 1.0
    if n_new is not None and n_cur is not None:
        mult *= 1 + k_n * math.log((1 + n_new) / (1 + n_cur))
    if comm_new is not None and comm_cur is not None and comm_cur > 0:
        mult *= 1 + k_comm * (comm_new / comm_cur - 1)
    return mult


def _predict_cell(
    model: ShareModel,
    cell: pd.DataFrame,
    company: int,
    sold_now: float | None,
    share_now: float | None,
    seas_ratio: float,
    price: float | None = None,
    ad_mult: float = 1.0,
    dist_mult: float = 1.0,
) -> tuple[float, float | None]:
    c = cell.copy()
    c["price"] = c["price"].astype(float)
    c["adspend"] = c["adspend"].astype(float)
    mask = c["company"] == company
    if price is not None:
        c.loc[mask, "price"] = float(price)
    if ad_mult != 1.0:
        c.loc[mask, "adspend"] *= ad_mult
    attr_mult: np.ndarray | None = None
    if dist_mult != 1.0:
        attr_mult = np.where(mask.to_numpy(), dist_mult, 1.0)
    shares = model.predict_shares(c, attr_mult=attr_mult)
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
    q_now = meta["quarter"]  # из текущего excel-файла (лист 'W')
    q_next = q_now % 4 + 1  # квартал прогноза — следующий за текущим
    seas_ratio = (seas[q_next] / seas[q_now]) if seas else 1.0
    price_sc_raw = scenario.get("price", {})
    price_sc: dict[str, float] = (
        price_sc_raw if isinstance(price_sc_raw, dict) else {}
    )
    ad_mult_raw = scenario.get("adspend_mult", 1.0)
    ad_mult = (
        float(ad_mult_raw) if isinstance(ad_mult_raw, (int, float)) else 1.0
    )
    dist_n_raw = scenario.get("dist_n", {})
    dist_n_sc: dict[str, float] = (
        dist_n_raw if isinstance(dist_n_raw, dict) else {}
    )
    dist_comm_raw = scenario.get("dist_comm", {})
    dist_comm_sc: dict[str, float] = (
        dist_comm_raw if isinstance(dist_comm_raw, dict) else {}
    )
    dist_el_raw = scenario.get("dist_elasticity", {})
    dist_el: dict[str, float] = (
        dist_el_raw if isinstance(dist_el_raw, dict) else {}
    )
    k_n = float(dist_el.get("n", DIST_K_N))
    k_comm = float(dist_el.get("comm", DIST_K_COMM))

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
            # дистрибьюторы — пер-канальные (одинаковы для 3 продуктов канала)
            dist_n_now = (
                float(orow["dist_n"]) if pd.notna(orow["dist_n"]) else None
            )
            dist_comm_now = (
                float(orow["dist_comm"])
                if pd.notna(orow["dist_comm"])
                else None
            )
            new_dist_n = dist_n_sc.get(ch, dist_n_now)
            new_dist_comm = dist_comm_sc.get(ch, dist_comm_now)
            dist_mult = _dist_mult(
                dist_n_now,
                new_dist_n,
                dist_comm_now,
                new_dist_comm,
                k_n,
                k_comm,
            )

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
                dist_mult=dist_mult,
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
                    "дистриб_текущ": dist_n_now,
                    "дистриб_сцен": new_dist_n,
                    "комис_текущ": dist_comm_now,
                    "комис_сцен": new_dist_comm,
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
