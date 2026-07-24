"""Тесты авто-выбора уровня доли (anchored ↔ absolute) по разрыву gap.

gap = набл_доля/доля_база. Авто-режим доверяет якорю ровно настолько, насколько
разрыв ПЕРСИСТЕНТЕН по своей истории (gap_persistence): устойчивый разрыв →
якорь, разрыв, схлопывающийся к 1 (рамп/вход) → сдвиг уровня к модельной доле.
Без истории (первый квартал) — откат на уровневый приор gap_weight.
"""

from __future__ import annotations
import glob
import math
from pathlib import Path
import pandas as pd
import pytest
from gmc_forecaster.model import (
    gap_weight,
    gap_persistence,
    anchor_weight,
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


# --- gap_persistence: авто-вес по ДИНАМИКЕ разрыва --------------------------
def test_persistence_no_history_falls_back_to_prior() -> None:
    """Меньше двух разрывов → чистый уровневый приор gap_weight(текущий)."""
    assert gap_persistence([0.3]) == pytest.approx(gap_weight(0.3))
    assert gap_persistence([]) == 1.0


def test_persistence_stable_gap_trusts_anchor() -> None:
    """Устойчивый разрыв 0.3 (не рамп) → w высокий (якорь), несмотря на то что
    уровневый приор в одиночку дал бы низкий вес."""
    w = gap_persistence([0.30, 0.28, 0.31, 0.30])
    assert w > 0.9
    assert w > gap_weight(0.30)  # динамика бьёт уровневый приор


def test_persistence_ramp_trusts_model() -> None:
    """Рамп: разрыв летит к 1 (0.3→1.2→1.1) → w низкий (модель)."""
    assert gap_persistence([0.32, 1.21, 1.09]) < 0.3


def test_persistence_clipped_to_unit_interval() -> None:
    """w клипуется в [0,1]: усиление разрыва → 1, перескок через 1 → 0."""
    assert gap_persistence([2.0, 4.0, 8.0]) == pytest.approx(1.0, abs=0.05)
    assert 0.0 <= gap_persistence([0.2, 5.0]) <= 1.0


def test_anchor_weight_uses_history() -> None:
    """anchor_weight(auto) с историей устойчивого разрыва → якорь; без истории
    тот же текущий разрыв даёт низкий вес (уровневый приор)."""
    hist = [0.30, 0.28, 0.31]
    w_hist = anchor_weight(0.9, 3.0, "auto", gap_hist=hist)  # gap=0.3
    w_bare = anchor_weight(0.9, 3.0, "auto")
    assert w_hist > 0.9 > w_bare


def test_anchor_weight_modes() -> None:
    assert anchor_weight(1.0, 5.0, "anchored") == 1.0
    assert anchor_weight(1.0, 5.0, "absolute") == 0.0
    assert anchor_weight(None, 5.0, "auto") == 1.0


def test_level_factor_history_stabilizes_persistent_gap() -> None:
    """Устойчивый разрыв 0.3 с историей → фактор ≈ 1 (якорь), тогда как без
    истории тот же разрыв поднимал уровень к модели (фактор ≫ 1)."""
    f_hist = level_factor(0.9, 3.0, "auto", gap_hist=[0.30, 0.28, 0.31])
    f_bare = level_factor(0.9, 3.0, "auto")
    assert f_hist == pytest.approx(1.0, abs=0.15)
    assert f_bare > 2.0


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


# --- извлечение истории разрыва: gap_panel / gaps_before --------------------
# гр.15 к.2 — зрелая фирма с УСТОЙЧИВО слабой АСЕАН (разрыв ≈0.3, не рамп).
# train включает свою серию (W172) — как в реальной команде backtest.
G15 = glob.glob(str(ROOT / "data" / "W152*.xls"))
TRAIN15 = TRAIN + glob.glob(str(ROOT / "data" / "W172*.xls"))


@pytest.mark.skipif(not (G15 and TRAIN15), reason="нет отчётов гр.15")
def test_gap_panel_and_gaps_before() -> None:
    """gap_panel собирает хронологию разрыва по своей серии; gaps_before
    отдаёт кварталы СТРОГО раньше t (защита бэктеста от заглядывания вперёд)."""
    from gmc_forecaster.model import (
        ShareModel,
        load_panel,
        gap_panel,
        gaps_before,
    )

    model = ShareModel().fit(load_panel(TRAIN15))
    panel = gap_panel(model, G15)
    assert not panel.empty
    assert set(panel.columns) == {"group", "company", "t", "cell", "gap"}
    # гр.15 к.2 АСЕАН3 — устойчиво слабая ячейка: все разрывы заметно <1
    a3 = panel[(panel["cell"] == "ASEAN3") & (panel["company"] == 2)]
    assert (a3["gap"] < 0.7).all()
    # отсечка по времени: до самого раннего квартала истории нет
    tmin = int(panel["t"].min())
    assert gaps_before(panel, 15, 2, tmin) == {}
    before_last = gaps_before(panel, 15, 2, int(panel["t"].max()))
    assert len(before_last.get("ASEAN3", [])) >= 1


@pytest.mark.skipif(not (G15 and TRAIN15), reason="нет отчётов гр.15")
def test_backtest_history_fixes_persistent_gap_cell() -> None:
    """Регрессия: устойчиво слабая ASEAN3 гр.15 (разрыв ≈0.3 три квартала) при
    авто-режиме с историей заякоривается, а не поднимается к модели в ×3.
    Прежний уровневый auto давал спрос-MAPE по этой ячейке ~148%, а по агрегату
    проваливался ниже сезонного наива (32.8% против 20.6%)."""
    from gmc_forecaster.backtest import backtest, per_cell_summary

    summ, df = backtest(G15, TRAIN15, [], mode="auto", seasonality="market")
    assert summ["история_разрыва_кв"] >= 2
    pc = per_cell_summary(df)
    a3 = pc[(pc["канал"] == "ASEAN") & (pc["продукт"] == 3)].iloc[0]
    # заякоренный уровень → ошибка спроса умеренная (прежний auto: ~148%)
    assert float(a3["спрос_MAPE_real"]) < 30.0
    # авто с историей ≈ anchored и заметно ниже прежнего уровневого auto
    anc = backtest(G15, TRAIN15, [], mode="anchored", seasonality="market")[0]
    assert summ["спрос_MAPE_real"] < 1.15 * anc["спрос_MAPE_real"]
    # агрегат больше не проваливается ниже сезонного наива
    assert summ["спрос_MAPE_real"] < summ["спрос_MAPE_seasnaive"]
