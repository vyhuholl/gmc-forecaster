"""Тесты авто-выбора уровня доли (anchored ↔ absolute) по разрыву gap.

gap = набл_доля/доля_база: gap≈1 → якорь (наблюдаемый спрос персистентен),
gap далёк от 1 → фирма вне режима (рамп) → сдвиг уровня к модельной доле.
"""

from __future__ import annotations
import glob
import math
from pathlib import Path
import pandas as pd
import pytest
from gmc_forecaster.model import (
    gap_weight,
    level_factor,
    GAP_TAU,
    MODES,
)
from gmc_forecaster.forecast import forecast

ROOT = Path(__file__).resolve().parent.parent
CUR = str(ROOT / "data" / "W172263.xls")  # gr17 — рампующая фирма (АСЕАН)
TRAIN = (
    glob.glob(str(ROOT / "data" / "W115*.xls"))
    + glob.glob(str(ROOT / "data" / "W131*.xls"))
    + glob.glob(str(ROOT / "data" / "W318*.xls"))
    + glob.glob(str(ROOT / "data" / "W496*.xls"))
)


# --- gap_weight: гауссиана по ln(gap) ---------------------------------------
def test_gap_weight_peak_at_one() -> None:
    """w(1)=1 (равновесие → полное доверие якорю)."""
    assert gap_weight(1.0) == pytest.approx(1.0)


def test_gap_weight_decreases_away_from_one() -> None:
    """w убывает по мере отклонения gap от 1 в обе стороны."""
    assert gap_weight(1.0) > gap_weight(0.5) > gap_weight(0.15)
    assert gap_weight(1.0) > gap_weight(2.0) > gap_weight(6.0)


def test_gap_weight_log_symmetric() -> None:
    """Симметрия в лог-долях: w(gap)=w(1/gap)."""
    assert gap_weight(0.3) == pytest.approx(gap_weight(1 / 0.3))


def test_gap_weight_tau_monotone() -> None:
    """Больший tau → ближе к якорю (w выше) при том же разрыве."""
    assert gap_weight(0.4, tau=1.0) > gap_weight(0.4, tau=0.3)


@pytest.mark.parametrize("bad", [None, 0.0, -1.0])
def test_gap_weight_guards(bad: float | None) -> None:
    assert gap_weight(bad) == 1.0


# --- level_factor: множитель к заякоренной базе ------------------------------
def test_level_factor_anchored_is_identity() -> None:
    """anchored → 1.0 при любом разрыве (якорь как есть)."""
    assert level_factor(1.0, 6.8, "anchored") == 1.0


def test_level_factor_absolute_maps_to_model_share() -> None:
    """absolute → доля_база/набл: перевод наблюдаемого уровня к модельному."""
    # спрос_база×factor = набл_доля×(доля_база/набл)×рынок ∝ доля_база
    assert level_factor(1.0, 6.8, "absolute") == pytest.approx(6.8)


def test_level_factor_auto_neutral_at_equilibrium() -> None:
    """auto при gap=1 → фактор ≈ 1 (не трогает равновесную ячейку)."""
    assert level_factor(4.0, 4.0, "auto") == pytest.approx(1.0)


def test_level_factor_auto_between_anchored_and_absolute() -> None:
    """auto на рампе (gap<1) даёт фактор в (1, доля_база/набл)."""
    f = level_factor(1.0, 6.8, "auto")
    assert 1.0 < f < 6.8


@pytest.mark.parametrize(
    "sn,sb", [(None, 5.0), (5.0, None), (0.0, 5.0), (5.0, 0.0)]
)
def test_level_factor_guards_fall_back_to_anchor(
    sn: float | None, sb: float | None
) -> None:
    assert level_factor(sn, sb, "auto") == 1.0


def test_level_factor_auto_matches_manual_formula() -> None:
    """auto = gap^(w−1), w=gap_weight(gap)."""
    sn, sb = 1.0, 6.8
    gap = sn / sb
    w = gap_weight(gap, GAP_TAU)
    assert level_factor(sn, sb, "auto") == pytest.approx(gap ** (w - 1.0))


# --- интеграция: forecast на рампующей фирме (gr17) --------------------------
@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_forecast_modes_are_valid() -> None:
    assert set(MODES) == {"auto", "anchored", "absolute"}


def _demand(df: pd.DataFrame, ch: str, p: int) -> float:
    """спрос_сцен по ячейке (канал, продукт) как float."""
    row = df[(df["канал"] == ch) & (df["продукт"] == p)].iloc[0]
    return float(row["спрос_сцен"])


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_forecast_auto_lifts_ramp_cells_above_anchored() -> None:
    """На АСЕАН (рамп, gap≈0.15-0.30) auto и absolute поднимают спрос над
    anchored; на ЕАЭС (gap≈1) auto почти совпадает с anchored."""
    anc = forecast(CUR, TRAIN, [], mode="anchored")
    aut = forecast(CUR, TRAIN, [], mode="auto")
    ab = forecast(CUR, TRAIN, [], mode="absolute")
    # АСЕАН — фирма недобирает модельную долю → уровень сдвигается вверх
    for p in (1, 2, 3):
        assert _demand(aut, "АСЕАН", p) > _demand(anc, "АСЕАН", p)
        assert _demand(ab, "АСЕАН", p) >= _demand(aut, "АСЕАН", p)
    # ЕАЭС-1 в равновесии (gap≈0.93) → auto почти = anchored (±3%)
    a = _demand(anc, "ЕАЭС", 1)
    u = _demand(aut, "ЕАЭС", 1)
    assert abs(u - a) / a < 0.03


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_forecast_exposes_gap_and_weight() -> None:
    df = forecast(CUR, TRAIN, [], mode="auto")
    assert "разрыв_доли" in df.columns
    assert "вес_якоря" in df.columns
    # рампующая АСЕАН1: разрыв заметно <1, вес доверия якорю мал
    row = df[(df["канал"] == "АСЕАН") & (df["продукт"] == 1)].iloc[0]
    assert float(row["разрыв_доли"]) < 0.6
    assert float(row["вес_якоря"]) < 0.2
    # равновесная ЕАЭС2: разрыв ≈1, вес ≈1
    row = df[(df["канал"] == "ЕАЭС") & (df["продукт"] == 2)].iloc[0]
    assert abs(float(row["разрыв_доли"]) - 1.0) < 0.2
    assert float(row["вес_якоря"]) > 0.9


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_forecast_anchored_matches_legacy_no_level_shift() -> None:
    """anchored: вес_якоря=1 везде, спрос_база = спрос_текущ (seas=none)."""
    df = forecast(CUR, TRAIN, [], mode="anchored", seasonality="none")
    assert (df["вес_якоря"] == 1.0).all()
    assert (df["спрос_база"] == df["спрос_текущ"]).all()


def test_gap_weight_clip_matches_forecast_bound() -> None:
    """Экстремальный разрыв клипуется до GAP_CLIP=[0.1,10] в level_factor."""
    # gap=0.01 клипуется до 0.1 → фактор = 0.1^(w−1) с w=gap_weight(0.1)
    w = gap_weight(0.1)
    assert level_factor(0.05, 5.0, "auto") == pytest.approx(0.1 ** (w - 1.0))
    assert not math.isinf(level_factor(1e-6, 5.0, "auto"))
