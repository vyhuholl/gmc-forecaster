"""Тесты себестоимости: тождества расхода, баланс разнесения, EWMA, цены,
оба выхода, деградация в history-режиме."""

from __future__ import annotations
import dataclasses
import glob
import math
from pathlib import Path
import pytest
from gmc_forecaster.parser import parse_costs, parse_decisions, PRODUCTS
from gmc_forecaster.cost import (
    cost,
    unit_costs,
    raw_used,
    machine_hours,
    assembler_hours,
    calibrate_yields,
    material_unit_price,
    _ewma,
    _plan_edited,
    RAW_PER_UNIT,
    ASSEMBLY_MIN,
    CHANNELS,
)

ROOT = Path(__file__).resolve().parent.parent
W_FILES = sorted(glob.glob(str(ROOT / "data" / "W*.xls")))
HST = str(ROOT / "data" / "HstY26Q1.xlsx")
CUR = str(ROOT / "data" / "W115264.xls")
TRAIN = [str(ROOT / "data" / f"W11526{q}.xls") for q in (1, 2, 3)]


@pytest.mark.parametrize("path", W_FILES)
def test_raw_identity(path: str) -> None:
    """Сырьё_использ = Σ (произв−п/ф)×{1,2,3} совпадает с фактом до единицы."""
    c = parse_costs(path)
    if not 1 <= c.company <= 8:
        pytest.skip("не производственный отчёт")
    assert raw_used(c.produced, c.components_used) == pytest.approx(
        c.material_used, abs=1.0
    )


@pytest.mark.parametrize("path", W_FILES)
def test_machine_identity(path: str) -> None:
    """Станко-часы = Σ(произв−п/ф)×{60,75,120}/60/эфф ≈ факт (≤0.5 %)."""
    c = parse_costs(path)
    if not 1 <= c.company <= 8 or c.machine_efficiency <= 0:
        pytest.skip("нет эффективности станков")
    calc = machine_hours(c.produced, c.components_used, c.machine_efficiency)
    assert calc == pytest.approx(c.machine_worked_h, rel=0.005)


@pytest.mark.parametrize("path", W_FILES)
def test_overhead_checksum(path: str) -> None:
    """Сумма статей накладных == итоговая строка (контрольная сумма парсинга)."""
    c = parse_costs(path)
    assert sum(c.overhead.values()) == pytest.approx(c.overhead_total, abs=1.0)


def test_assembler_hours_norm() -> None:
    """Сборщико-часы = Σ произв × {100,150,300}/60 (п/ф сборку не замещают)."""
    c = parse_costs(CUR)
    expect = sum(c.produced[p] * ASSEMBLY_MIN[p] for p in PRODUCTS) / 60.0
    assert assembler_hours(c.produced) == pytest.approx(expect)


def test_allocation_balance() -> None:
    """Σ отнесённых по 9 ячейкам == пулы (полнота разнесения)."""
    d = cost(CUR, train=TRAIN)
    pools = d.attrs["pools"]
    assert d["материал"].sum() == pytest.approx(pools["материал"], rel=1e-6)
    assert d["конверсия"].sum() == pytest.approx(pools["конверсия"], rel=1e-6)
    assert d["дистриб"].sum() == pytest.approx(pools["канал"], rel=1e-6)
    assert d["накладные"].sum() == pytest.approx(pools["накладные"], rel=1e-6)
    assert d["себест_полн"].sum() == pytest.approx(pools["всего"], rel=1e-6)


def test_both_outputs() -> None:
    """Оба выхода присутствуют; прибыль_полн = цена − себест_полн_ед;
    CM не зависит от конверсии/накладных (= цена − материал − комиссия)."""
    d = cost(CUR, train=TRAIN)
    assert "себест_полн_ед" in d.columns and "CM_ед" in d.columns
    for _, r in d.iterrows():
        assert r["прибыль_полн_ед"] == pytest.approx(
            r["цена"] - r["себест_полн_ед"], abs=0.01
        )
        # CM = цена − материал/ед − комиссия/ед; конверсия/накладные исключены
        mat_pu = RAW_PER_UNIT[int(r["продукт"])] * d.attrs["meta"]["mat_price"]
        assert r["CM_ед"] == pytest.approx(
            r["цена"] - mat_pu - r["комиссия_ед"], abs=0.01
        )


def test_ewma_degenerate() -> None:
    """EWMA: пусто -> None, один элемент -> он сам (вырождение в точку)."""
    assert _ewma([], 2.0) is None
    assert _ewma([0.05], 2.0) == pytest.approx(0.05)


def test_yields_single_quarter() -> None:
    """Калибровка по одному кварталу не падает и даёт разумные множители."""
    c = parse_costs(CUR)
    y = calibrate_yields([c])
    assert y["defect_infl"] >= 1.0 and y["absence_infl"] >= 1.0
    assert 0.0 < y["efficiency"] <= 1.0


def test_material_price_zero_purchase() -> None:
    """Нулевая закупка сырья не даёт деления на ноль (фоллбэк на 0)."""
    c = parse_costs(CUR)
    zero = dataclasses.replace(
        c, material_purchased_units=0.0, material_purchased_cost=0.0
    )
    price = material_unit_price([zero], zero)
    assert price == 0.0 and math.isfinite(price)


def test_history_degradation() -> None:
    """history-рамп (компания 0): unit_costs -> None, cost -> ValueError."""
    assert unit_costs(HST) is None
    with pytest.raises(ValueError):
        cost(HST)


# ---------- вход себестоимости из РЕШЕНИЙ (change cost-from-decisions) ----------
def test_decisions_plan_equals_planned() -> None:
    """План поставок с листа решений (по продуктам) == planned отчёта."""
    dec = parse_decisions(CUR)
    c = parse_costs(CUR)
    by_prod = {
        p: sum(dec["plan"].get(f"{ch}{p}", 0.0) for ch in CHANNELS)
        for p in PRODUCTS
    }
    assert by_prod == pytest.approx(c.planned)


def test_decisions_assembly_time_reproduces_worked_h() -> None:
    """Время сборки С ЛИСТА РЕШЕНИЙ × выпуск воспроизводит assembler_worked_h
    (≤1 %), а константа мануала {100,150,300} — нет (расходится >5 %)."""
    dec = parse_decisions(CUR)
    c = parse_costs(CUR)
    asm = dec["assembly_min"]
    assert set(asm) == set(PRODUCTS)
    h_dec = sum(c.produced[p] * asm[p] for p in PRODUCTS) / 60.0
    h_const = sum(c.produced[p] * ASSEMBLY_MIN[p] for p in PRODUCTS) / 60.0
    assert h_dec == pytest.approx(c.assembler_worked_h, rel=0.01)
    assert abs(h_const / c.assembler_worked_h - 1.0) > 0.05


def test_cost_quantity_is_plan_not_sold() -> None:
    """Объём ячейки в смете = план поставок (не sold): напр. ASEAN2 = 70."""
    dec = parse_decisions(CUR)
    d = cost(CUR, train=TRAIN)
    for _, r in d.iterrows():
        ch = {"ЕАЭС": "EAEU", "АСЕАН": "ASEAN", "Интернет": "INT"}[
            str(r["канал"])
        ]
        key = f"{ch}{int(r['продукт'])}"
        assert r["ед"] == pytest.approx(dec["plan"][key])


def test_reconciliation_executed_within_threshold() -> None:
    """На предзаполненном листе (решения = исполненным) смета сходится с
    COGS+накл в пределах inventory-шума (≤5 %); режим = исполненные."""
    d = cost(CUR, train=TRAIN)
    pl = d.attrs["pl_check"]
    assert pl["применима"] is True
    assert d.attrs["meta"]["режим"] == "исполненные"
    gap = abs(pl["смета"] / pl["факт_cogs_плюс_накл"] - 1.0)
    assert gap <= 0.05


def test_forecast_base_columns_and_clip() -> None:
    """demand добавляет колонки базы прогноза; продано клипуется ≤ план;
    себест_полн_ед НЕ зависит от базы продаж."""
    base = cost(CUR, train=TRAIN)
    # EAEU1 план=1828 -> demand 1500 (ниже, берётся); INT1 план=1518 ->
    # demand 5000 (выше, клип к плану)
    demand = {"EAEU1": 1500.0, "INT1": 5000.0}
    d = cost(CUR, train=TRAIN, demand=demand)
    assert {"продано_прог", "выручка_прог", "прибыль_прог"} <= set(d.columns)
    row = d[(d["канал"] == "ЕАЭС") & (d["продукт"] == 1)].iloc[0]
    assert row["продано_прог"] == pytest.approx(1500.0)
    ir = d[(d["канал"] == "Интернет") & (d["продукт"] == 1)].iloc[0]
    assert ir["продано_прог"] == pytest.approx(ir["ед"])  # клип к плану
    # себест/ед базонезависима
    assert list(d["себест_полн_ед"]) == pytest.approx(
        list(base["себест_полн_ед"])
    )


def test_plan_edited_mode() -> None:
    """_plan_edited: нет плана/план=исполненному -> False; отклонение >2 % ->
    True (форвард-оценка)."""
    c = parse_costs(CUR)
    planned = c.planned
    assert _plan_edited({}, planned) is False
    same = {f"{ch}{p}": planned[p] / 3.0 for ch in CHANNELS for p in PRODUCTS}
    assert _plan_edited(same, planned) is False
    bumped = dict(same)
    bumped["EAEU1"] = planned[1]  # ~+66 % объёма продукта 1
    assert _plan_edited(bumped, planned) is True
