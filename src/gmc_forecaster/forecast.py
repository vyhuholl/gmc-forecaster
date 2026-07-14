"""
forecast.py — прогноз спроса на следующий квартал под решениями игрока.
Это рабочий интерфейс «крутить цены/рекламу»: меняешь свои решения на ПЕРВОМ
листе ('Your decisions') отчёта --current -> видишь спрос по всем 9 ячейкам
(3 продукта × 3 канала).

Пайплайн:
  1. Обучаем логит-модель доли на конкурентных отчётах (train).
  2. Оцениваем сезонность объёма по группе 0 (history, опционально).
  3. Берём текущее состояние рынка из current (все 8 компаний + свои продажи/доля).
  4. Читаем решения игрока с листа 'Your decisions' файла --current
     (parse_decisions) и применяем их к своей компании.
  5. По каждой ячейке ЯКОРИМ прогноз на наблюдаемый спрос:
     спрос_сцен = спрос_текущ × сезонность × рычаг, где рычаг = модельное
     отношение доли (сцен/база), демпфируемое lever_k. Так наблюдаемый спрос
     несёт всю firm×cell-гетерогенность (её логит доли не восстанавливает —
     оттого абсолютный прогноз доля×объём давал 15-20% MAPE), а модель отвечает
     лишь за причинный сдвиг от смены решений. При lever_k=0 -> сезонный наив.

Решения (цены/реклама/дистрибьюторы) больше НЕ подаются в json — они читаются
прямо из первого листа --current (см. parser.parse_decisions). База сравнения
(спрос_база, Δ_рычаг) — исходные решения квартала с листа 'W'; правишь форму
решений -> Δ_рычаг показывает эффект. Дистрибьюторы — пер-канальные рычаги
(EAEU/ASEAN/INT, действуют на все 3 продукта канала), наблюдаемы только у своей
компании -> дают множитель СОБСТВЕННОЙ привлекательности A_own в логите доли
(конкуренты не затронуты). Сила эффекта — коэффициенты k_n/k_comm (дефолты
DIST_K_N/DIST_K_COMM, опц. оверрайд из CLI). НАЙМ ОТЛОЖЕН НА КВАРТАЛ: выходит на
охват лишь со следующего за прогнозным квартала (Q_next+1), поэтому на прогноз
Q_next дистрибьюторы НЕ влияют (в отличие от цены/рекламы, действующих сразу).
Их отложенный вклад показывается отдельной колонкой Δ_дистриб_след_%. Квартал
прогноза берётся из текущего excel-файла (следующий за кварталом на листе 'W').

РЕЖИМ 1-Й ИТЕРАЦИИ (half-final, _forecast_history): если --current — history-
рамп (компания вне 1..8, своих долей нет: доли 'Not requested', 8 «компаний»
листа 'W' — идентичные клоны), конкурентную стадию 1 запустить нельзя. Тогда
forecast() авто-переключается: бейзлайн = ПЕРСИСТЕНЦИЯ наблюдаемого спроса по
ячейке, а рычаг решений — ЗАИМСТВОВАННАЯ собственная эластичность из модели,
обученной на регулярных отчётах (--train с долями): Δln(спрос) ≈ β_price·
Δln(цена) + β_adspend·Δln(реклама+1), демпфируется lever_k (дистрибьюторы
отложены на квартал -> в рычаг Q_next не входят). Это
грубый перенос из чужого контекста -> низкая уверенность (df.attrs['meta']
['mode']=='history'); доля_сцен_% не оценивается.
"""

from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd
from .parser import parse_report, parse_decisions
from .model import (
    load_panel,
    ShareModel,
    fit_seasonality,
    cell_volume,
    damp_lever,
    LEVER_K,
    CH,
)

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
    Пер-канальный рычаг: одинаков для всех 3 продуктов канала. Эффект найма
    ОТЛОЖЕН на квартал -> в рычаг прогнозного Q_next не входит, показывается
    отдельно (Δ_дистриб_след_%)."""
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
    current: str,
    train: list[str],
    history: list[str],
    k_n: float = DIST_K_N,
    k_comm: float = DIST_K_COMM,
    lever_k: float = LEVER_K,
) -> pd.DataFrame:
    model = ShareModel().fit(load_panel(train))
    seas = fit_seasonality(history) if history else None

    meta, own = parse_report(current)
    company = meta["company"]
    q_now = meta["quarter"]  # из текущего excel-файла (лист 'W')
    q_next = q_now % 4 + 1  # квартал прогноза — следующий за текущим
    seas_ratio = (seas[q_next] / seas[q_now]) if seas else 1.0
    # 1-я итерация half-final: свой отчёт с долями ещё не сыгран — на руках лишь
    # history-рамп (группа 0, компания 0, доли 'Not requested'). Конкурентную
    # стадию 1 запустить нельзя -> бейзлайн спроса + ЗАИМСТВОВАННЫЙ рычаг
    # (эластичность из --train); см. _forecast_history.
    if not 1 <= company <= 8:
        return _forecast_history(
            current,
            own,
            model,
            meta,
            q_now,
            q_next,
            seas_ratio,
            k_n,
            k_comm,
            lever_k,
        )

    cur_panel = load_panel([current])

    # решения игрока с первого листа 'Your decisions' файла --current
    dec = parse_decisions(current)
    price_sc: dict[str, float] = dec["price"]
    dist_n_sc: dict[str, float] = dec["dist_n"]
    dist_comm_sc: dict[str, float] = dec["dist_comm"]
    # реклама компании-уровня -> множитель относительно исходного бюджета из
    # панели (W[adspend_base] в ерз; adspend_total на листе — в тыс. ерз)
    orig_ad_ser = cur_panel.loc[cur_panel["company"] == company, "adspend"]
    orig_ad = (
        float(orig_ad_ser.iloc[0])
        if len(orig_ad_ser) and pd.notna(orig_ad_ser.iloc[0])
        else None
    )
    ad_total = float(dec["adspend_total"])
    ad_mult = ad_total * 1000.0 / orig_ad if orig_ad and orig_ad > 0 else 1.0

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
            # эффект дистрибьюторов ОТЛОЖЕН на квартал: найм выходит на охват
            # лишь со следующего за прогнозным квартала (Q_next+1). На прогноз
            # Q_next он НЕ влияет — держим текущий охват; вклад показываем
            # отдельной колонкой Δ_дистриб_след_%.
            dist_mult = _dist_mult(
                dist_n_now,
                new_dist_n,
                dist_comm_now,
                new_dist_comm,
                k_n,
                k_comm,
            )

            # доли модели: база (текущие решения) и сценарий (новые решения);
            # абсолютный спрос базы нужен лишь для отката, если спроса нет
            sh_base, d_base_abs = _predict_cell(
                model,
                cell,
                company,
                sold_now,
                share_now,
                seas_ratio,
                price=price_now,
                ad_mult=1.0,
            )
            # сценарий ТЕКУЩЕГО квартала: цена и реклама действуют сразу,
            # дистрибьюторы — нет (dist_mult не подаём)
            sh_sc, _ = _predict_cell(
                model,
                cell,
                company,
                sold_now,
                share_now,
                seas_ratio,
                price=new_price,
                ad_mult=ad_mult,
            )
            # причинный рычаг = модельное отношение доли сцен/база (=1, если
            # цену/рекламу не меняли), демпфированное lever_k и клипованное
            lever = damp_lever(sh_sc / sh_base if sh_base else 1.0, lever_k)
            # ОТЛОЖЕННЫЙ рычаг дистрибьюторов = их доп.вклад поверх сцен-доли:
            # то, что найм прибавит к спросу со СЛЕДУЮЩЕГО квартала
            dist_lever = 1.0
            if dist_mult != 1.0:
                sh_defer, _ = _predict_cell(
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
                dist_lever = damp_lever(
                    sh_defer / sh_sc if sh_sc else 1.0, lever_k
                )
            demand_now = (
                float(orow["demand"]) if pd.notna(orow["demand"]) else None
            )
            # ЯКОРЬ: база = наблюдаемый спрос × сезонность (сильный сез.-наив
            # бейзлайн), сценарий = база × рычаг. Наблюдаемый спрос несёт всю
            # firm×cell-гетерогенность, которую логит доли не восстанавливает.
            # Нет наблюдаемого спроса -> откат на абсолютный прогноз доля×объём.
            d_base: float | None
            if demand_now is not None:
                d_base = round(demand_now * seas_ratio)
            else:
                d_base = d_base_abs
            d_sc = round(d_base * lever) if d_base is not None else None
            delta = round((lever - 1.0) * 100, 1)
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
                    "Δ_дистриб_след_%": round((dist_lever - 1.0) * 100, 1),
                }
            )
    df = pd.DataFrame(out)
    df.attrs["meta"] = dict(
        company=company,
        group=meta["group"],
        q_now=q_now,
        q_next=q_next,
        seas_ratio=round(seas_ratio, 3),
        lever_k=lever_k,
    )
    # диагностика стадии 1: коэффициенты доли + значимость и качество подгонки
    df.attrs["coef_summary"] = model.coef_summary()
    df.attrs["fit"] = dict(
        r2=round(model.r2, 3),
        n=model.n,
        edf=round(model.edf_, 1),
        ridge=model.ridge_,
    )
    return df


def _borrow_lever(
    model: ShareModel,
    dln_price: float,
    dln_ad: float,
    dist_mult: float,
    lever_k: float,
) -> float:
    """ЗАИМСТВОВАННЫЙ рычаг решений для 1-й итерации half-final (своих долей и
    конкурентной ячейки нет — 8 «компаний» листа 'W' идентичные клоны, поэтому
    полноценный MNL-рычаг доли недоступен). Реакцию СВОЕГО спроса на СВОИ
    решения приближаем полулог-эластичностью логита, обученного на регулярных
    отчётах (--train), в малодольном/фикс.-рынок приближении (1−s)≈1:
        Δln(спрос) ≈ β_price·Δln(цена) + β_adspend·Δln(реклама+1) + ln(dist_mult)
    Итог демпфируется lever_k и клипуется (damp_lever). Это грубый перенос из
    чужого контекста — помечается низкой уверенностью."""
    raw = (
        math.exp(
            model.coef_["log_price"] * dln_price
            + model.coef_["log_adspend"] * dln_ad
        )
        * dist_mult
    )
    return damp_lever(raw, lever_k)


def _forecast_history(
    current: str,
    own: pd.DataFrame,
    model: ShareModel,
    meta: dict[str, Any],
    q_now: int,
    q_next: int,
    seas_ratio: float,
    k_n: float,
    k_comm: float,
    lever_k: float,
) -> pd.DataFrame:
    """Прогноз 1-й итерации half-final: только history-рамп своей компании, БЕЗ
    долей рынка. Бейзлайн спроса = ПЕРСИСТЕНЦИЯ наблюдаемого спроса по ячейке
    (последний квартал × сезонность, если задана; сезонность из 5 рамп-кварталов
    ненадёжна -> по умолчанию 1). Рычаг решений — заимствованная эластичность
    (_borrow_lever). База решений — исполненные значения листа 'W' (у клонов
    идентичны), сценарий — с листа 'Ваши решения' (parse_decisions). Стадия 1
    (конкурентная доля) недоступна -> доля_сцен_% = None."""
    cur_panel = load_panel([current])
    dec = parse_decisions(current)
    price_sc: dict[str, float] = dec["price"]
    dist_n_sc: dict[str, float] = dec["dist_n"]
    dist_comm_sc: dict[str, float] = dec["dist_comm"]
    # база рекламы: клоны идентичны -> уровень компании из панели (ерз);
    # adspend_total листа — в тыс. ерз
    ad_ser = cur_panel["adspend"].dropna()
    base_ad = float(ad_ser.iloc[0]) if len(ad_ser) else None
    scen_ad = float(dec["adspend_total"]) * 1000.0
    dln_ad = (
        math.log((scen_ad + 1.0) / (base_ad + 1.0))
        if base_ad is not None and base_ad >= 0
        else 0.0
    )

    out = []
    for ch in CH:
        for p in (1, 2, 3):
            key = f"{ch}{p}"
            orow = own[(own["channel"] == ch) & (own["product"] == p)].iloc[0]
            demand_now = (
                float(orow["demand"]) if pd.notna(orow["demand"]) else None
            )
            # своя цена (company=0) в 'W' пуста -> берём из клона-панели ячейки
            cell_prices = cur_panel.loc[
                (cur_panel["channel"] == ch) & (cur_panel["product"] == p),
                "price",
            ].dropna()
            base_price = (
                float(cell_prices.median()) if len(cell_prices) else None
            )
            new_price = price_sc.get(key, base_price)
            dln_price = (
                math.log(new_price / base_price)
                if base_price and new_price and base_price > 0
                else 0.0
            )
            # дистрибьюторы — пер-канальные, база из 'W' (не завязана на company)
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
            # дистрибьюторы ОТЛОЖЕНЫ на квартал -> в рычаг Q_next не входят
            # (dist_mult=1.0); их отложенный вклад считаем отдельно
            lever = _borrow_lever(model, dln_price, dln_ad, 1.0, lever_k)
            dist_lever = (
                damp_lever(dist_mult, lever_k) if dist_mult != 1.0 else 1.0
            )
            # ЯКОРЬ: база = наблюдаемый спрос × сезонность; сценарий = база ×
            # заимствованный рычаг. Δ_рычаг изолирует эффект своих решений.
            d_base = (
                round(demand_now * seas_ratio)
                if demand_now is not None
                else None
            )
            d_sc = round(d_base * lever) if d_base is not None else None
            delta = round((lever - 1.0) * 100, 1)
            out.append(
                {
                    "канал": RU[ch],
                    "продукт": p,
                    "спрос_текущ": int(demand_now)
                    if demand_now is not None
                    else None,
                    "цена_текущ": base_price,
                    "цена_сцен": new_price,
                    "дистриб_текущ": dist_n_now,
                    "дистриб_сцен": new_dist_n,
                    "комис_текущ": dist_comm_now,
                    "комис_сцен": new_dist_comm,
                    "доля_сцен_%": None,  # долей нет -> стадия 1 недоступна
                    "спрос_база": d_base,
                    "спрос_сцен": d_sc,
                    "Δ_рычаг_%": delta,
                    "Δ_дистриб_след_%": round((dist_lever - 1.0) * 100, 1),
                }
            )
    df = pd.DataFrame(out)
    df.attrs["meta"] = dict(
        company=meta["company"],
        group=meta["group"],
        q_now=q_now,
        q_next=q_next,
        seas_ratio=round(seas_ratio, 3),
        lever_k=lever_k,
        mode="history",  # 1-я итерация: бейзлайн + заимствованный рычаг
    )
    # коэффициенты = ЗАИМСТВОВАННЫЕ эластичности (из --train), не свои доли
    df.attrs["coef_summary"] = model.coef_summary()
    df.attrs["fit"] = dict(
        r2=round(model.r2, 3),
        n=model.n,
        edf=round(model.edf_, 1),
        ridge=model.ridge_,
    )
    return df
