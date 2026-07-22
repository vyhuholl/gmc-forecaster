"""Тесты отчётов без рекламы/рейтинга (так устроен финальный отчёт партии:
цены и доли выгружены, а adspend/rating пусты у всех 8 компаний).

Три вещи, которые ломались: панель из ОДНОГО такого файла получала dtype=object
и роняла np.log в _design; подстановка нуля вместо пропуска обваливала ячейку во
внешнюю опцию (логит Берри не ренормирует доли к 100); те же нули втихую
обучали β_adspend на выдуманном «рекламы не было».
"""

from __future__ import annotations
import glob
from pathlib import Path
import pandas as pd
import pytest
from gmc_forecaster.backtest import backtest, _carry_forward
from gmc_forecaster.model import load_panel, ShareModel

ROOT = Path(__file__).resolve().parent.parent
NO_AD = str(ROOT / "data" / "W172272.xls")  # финальный отчёт: рекламы нет
PREV = str(ROOT / "data" / "W172271.xls")  # предыдущий квартал той же серии
OTHERS = sorted(
    glob.glob(str(ROOT / "data" / "W115*.xls"))
    + glob.glob(str(ROOT / "data" / "W131*.xls"))
    + glob.glob(str(ROOT / "data" / "W318*.xls"))
    + glob.glob(str(ROOT / "data" / "W496*.xls"))
)


def _cell(panel: pd.DataFrame) -> pd.DataFrame:
    return panel[(panel["channel"] == "EAEU") & (panel["product"] == 1)]


# --- load_panel: числовые типы независимо от полноты файла -------------------
def test_panel_numeric_even_if_column_empty() -> None:
    """Панель из одного файла без рекламы: колонки числовые, значения NaN."""
    p = load_panel([NO_AD])
    for c in ("price", "share", "adspend", "rating"):
        assert pd.api.types.is_numeric_dtype(p[c]), c
    assert p["adspend"].isna().all()
    assert p["price"].notna().all()  # цены при этом на месте


# --- предсказание: пропуск ≠ ноль -------------------------------------------
def test_predict_does_not_collapse_cell() -> None:
    """Без рекламы доли ячейки остаются правдоподобными, а не уезжают в s₀.
    С прежним fillna(0) выходило 0.02% при факте 12% (сумма по ячейке 0.09%)."""
    model = ShareModel().fit(load_panel([PREV] + OTHERS))
    shares = model.predict_shares(_cell(load_panel([NO_AD])))
    assert shares.sum() > 20.0
    assert shares.max() < 100.0


def test_missing_adspend_is_common_shift() -> None:
    """Пропуск заменяется одним и тем же средним у всех компаний ячейки, т.е.
    не перетасовывает их относительный порядок."""
    model = ShareModel().fit(load_panel([PREV] + OTHERS))
    cell = _cell(load_panel([NO_AD]))
    miss = model.predict_shares(cell)
    known = model.predict_shares(cell.assign(adspend=360000.0))
    assert list(miss.argsort()) == list(known.argsort())


# --- фит: строки без рекламы не учат модель нулям ---------------------------
def test_fit_excludes_rows_without_adspend() -> None:
    """Строки без рекламы исключены из обучения и посчитаны."""
    model = ShareModel().fit(load_panel([PREV, NO_AD] + OTHERS))
    clean = ShareModel().fit(load_panel([PREV] + OTHERS))
    assert model.n_no_adspend == 72  # 8 компаний × 3 продукта × 3 канала
    assert model.n == clean.n
    assert model.coef_["log_adspend"] == pytest.approx(
        clean.coef_["log_adspend"]
    )


def test_fit_survives_when_adspend_never_known() -> None:
    """Если рекламы нет НИГДЕ, фильтр не применяется — обучать иначе не на чем."""
    model = ShareModel().fit(load_panel([NO_AD]))
    assert model.n > 0


# --- backtest: пара с таким Q_next считается, а не падает -------------------
def test_carry_forward_fills_from_previous_quarter() -> None:
    cur, nxt = _cell(load_panel([PREV])), _cell(load_panel([NO_AD]))
    filled, carried = _carry_forward(nxt, cur)
    assert carried
    assert filled["adspend"].notna().all()
    assert filled["rating"].notna().all()
    # цены остаются фактическими, Q_next
    assert filled["price"].tolist() == nxt["price"].tolist()


def test_carry_forward_noop_when_complete() -> None:
    cur, nxt = _cell(load_panel([PREV])), _cell(load_panel([PREV]))
    filled, carried = _carry_forward(nxt, cur)
    assert not carried
    assert filled["adspend"].tolist() == nxt["adspend"].tolist()


def test_backtest_handles_pair_with_incomplete_next() -> None:
    """Пара 27Q1→27Q2 считается: раньше здесь падал oracle-блок."""
    summary, df = backtest([PREV, NO_AD] + OTHERS)
    assert summary["ячеек_реклама_с_Q_now"] == 9
    assert summary["строк_вне_фита_без_рекламы"] == 72
    pair = df[(df["группа"] == 17) & (df["Q_next"] == 2)]
    assert len(pair) == 9
    assert pair["доля_oracle"].between(1.0, 30.0).all()
    assert pair["спрос_real"].notna().all()
