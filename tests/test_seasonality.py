"""Тесты источника сезонности: рынок из конкурентных отчётов vs группа-0.

Рынок ячейки = свой_спрос/(своя_доля/100) убирает вариацию своей доли →
динамика рынка (сезон+тренд). В half-final группа-0 --history — это РАМП входа
по каналам, не сезонность (ложный Q3-пик); train-рынок 2026 — верный источник.
"""

from __future__ import annotations
import glob
from pathlib import Path
import pytest
from gmc_forecaster.model import (
    fit_market_seasonality,
    resolve_seasonality,
    SEASONALITY,
)
from gmc_forecaster.forecast import forecast

ROOT = Path(__file__).resolve().parent.parent
CUR = str(ROOT / "data" / "W172263.xls")
TRAIN = (
    glob.glob(str(ROOT / "data" / "W115*.xls"))
    + glob.glob(str(ROOT / "data" / "W131*.xls"))
    + glob.glob(str(ROOT / "data" / "W318*.xls"))
    + glob.glob(str(ROOT / "data" / "W496*.xls"))
)
HIST = sorted(glob.glob(str(ROOT / "data" / "half-final" / "Hst*.xlsx")))


# --- fit_market_seasonality -------------------------------------------------
@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_market_seasonality_shape() -> None:
    s = fit_market_seasonality(TRAIN)
    assert set(s) == {1, 2, 3, 4}
    gm = 1.0
    for v in s.values():
        gm *= v
    assert gm == pytest.approx(1.0, abs=1e-6)  # геомеан 1


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_market_seasonality_q4_above_q3() -> None:
    """Рынок растёт в Q4 относительно Q3 (сезонный пик) — >+10%."""
    s = fit_market_seasonality(TRAIN)
    assert s[4] / s[3] > 1.1


def test_market_seasonality_empty() -> None:
    assert fit_market_seasonality([]) == {}


# --- resolve_seasonality ----------------------------------------------------
def test_seasonality_choices() -> None:
    assert set(SEASONALITY) == {"auto", "market", "history", "none"}


def test_resolve_none() -> None:
    assert resolve_seasonality("none", TRAIN, HIST) == (None, "none")


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_resolve_market_from_train() -> None:
    seas, src = resolve_seasonality("market", TRAIN, HIST)
    assert src == "market" and seas is not None


@pytest.mark.skipif(not HIST, reason="нет history")
def test_resolve_history() -> None:
    seas, src = resolve_seasonality("history", TRAIN, HIST)
    assert src == "history" and seas is not None


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_resolve_auto_prefers_market() -> None:
    """auto при наличии train → market (не рамповая half-final history)."""
    assert resolve_seasonality("auto", TRAIN, HIST)[1] == "market"


@pytest.mark.skipif(not HIST, reason="нет history")
def test_resolve_auto_falls_back_to_history() -> None:
    assert resolve_seasonality("auto", [], HIST)[1] == "history"


def test_resolve_auto_none_when_nothing() -> None:
    assert resolve_seasonality("auto", [], []) == (None, "none")


def test_resolve_market_no_train_is_none() -> None:
    assert resolve_seasonality("market", [], HIST) == (None, "none")


# --- интеграция: сезонность двигает уровень прогноза ------------------------
def _demand(df: object, ch: str, p: int) -> float:
    import pandas as pd

    assert isinstance(df, pd.DataFrame)
    row = df[(df["канал"] == ch) & (df["продукт"] == p)].iloc[0]
    return float(row["спрос_сцен"])


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_forecast_market_lifts_over_none_in_growing_quarter() -> None:
    """Q3→Q4 рынок растёт → market-сезонность поднимает спрос над none."""
    none_df = forecast(CUR, TRAIN, [], seasonality="none")
    mkt_df = forecast(CUR, TRAIN, [], seasonality="market")
    assert mkt_df.attrs["meta"]["seasonality"] == "market"
    assert none_df.attrs["meta"]["seasonality"] == "none"
    # ЕАЭС3 в равновесии (нет сдвига уровня) → чистый сезонный подъём
    assert _demand(mkt_df, "ЕАЭС", 3) > _demand(none_df, "ЕАЭС", 3)


@pytest.mark.skipif(not TRAIN, reason="нет обучающих отчётов")
def test_forecast_auto_uses_market_when_train_present() -> None:
    df = forecast(CUR, TRAIN, HIST, seasonality="auto")
    assert df.attrs["meta"]["seasonality"] == "market"
