"""Тесты себестоимости: тождества расхода, баланс разнесения, EWMA, цены,
оба выхода, деградация в history-режиме."""

from __future__ import annotations
import dataclasses
import glob
import math
from pathlib import Path
import pytest
from gmc_forecaster.parser import parse_costs, PRODUCTS
from gmc_forecaster.cost import (
    cost,
    unit_costs,
    raw_used,
    machine_hours,
    assembler_hours,
    calibrate_yields,
    material_unit_price,
    _ewma,
    RAW_PER_UNIT,
    ASSEMBLY_MIN,
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
