"""
cli.py — точка входа `gmc-forecaster`. Подкоманды:
  • forecast — прогноз спроса на следующий квартал под сценарием решений;
  • backtest — оценка качества модели на исторических данных (без сценария);
  • cost     — полная себестоимость и contribution margin по 9 ячейкам.
"""

from __future__ import annotations
import argparse
import math
import os
import sys
from typing import Any
import pandas as pd
from .forecast import forecast, DIST_K_N, DIST_K_COMM
from .model import LEVER_K
from .backtest import backtest, per_cell_summary, _fin
from .cost import cost, YIELD_HALFLIFE


def _add_forecast(sub: argparse._SubParsersAction[Any]) -> None:
    fc = sub.add_parser(
        "forecast",
        help="прогноз спроса; решения — с листа 'Your decisions' файла "
        "--current",
    )
    fc.add_argument(
        "--current",
        required=True,
        help="текущий отчёт (.xls/.xlsx); решения читаются с его первого "
        "листа 'Your decisions'",
    )
    fc.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="конкурентные отчёты для обучения",
    )
    fc.add_argument(
        "--history",
        nargs="*",
        default=[],
        help="файлы группы 0 для сезонности",
    )
    fc.add_argument(
        "--dist-k-n",
        type=float,
        default=DIST_K_N,
        help=f"коэф. эффекта числа дистрибьюторов (дефолт {DIST_K_N})",
    )
    fc.add_argument(
        "--dist-k-comm",
        type=float,
        default=DIST_K_COMM,
        help=f"коэф. эффекта комиссии дистрибьюторов (дефолт {DIST_K_COMM})",
    )
    fc.add_argument(
        "--lever-k",
        type=float,
        default=LEVER_K,
        help=f"демпфер причинного рычага: спрос_сцен = спрос_база × "
        f"[1+(рычаг−1)·k] (дефолт {LEVER_K}; 0 = чистый сезонный наив)",
    )
    fc.add_argument("--out", help="сохранить прогноз в CSV")
    fc.add_argument(
        "--coeffs",
        choices=["key", "full", "none"],
        default="key",
        help="коэффициенты модели доли: key (ключевые+наклоны), "
        "full (+FE), none",
    )
    fc.set_defaults(func=_cmd_forecast)


def _add_backtest(sub: argparse._SubParsersAction[Any]) -> None:
    bt = sub.add_parser(
        "backtest",
        help="оценка качества модели на истории (пары смежных кварталов)",
    )
    bt.add_argument(
        "--reports",
        nargs="+",
        required=True,
        help="отчёты для оценки; пары смежных кварталов одной серии",
    )
    bt.add_argument(
        "--train",
        nargs="*",
        default=[],
        help="файлы для обучения модели доли (дефолт = --reports)",
    )
    bt.add_argument(
        "--history",
        nargs="*",
        default=[],
        help="файлы группы 0 для сезонности",
    )
    bt.add_argument(
        "--lever-k",
        type=float,
        default=LEVER_K,
        help=f"демпфер причинного рычага заякоренного прогноза "
        f"(дефолт {LEVER_K}; 1 = полный рычаг, 0 = чистый сезонный наив)",
    )
    bt.add_argument("--out", help="сохранить детализацию по ячейкам в CSV")
    bt.add_argument(
        "--per-cell",
        action="store_true",
        help="печатать метрики по 9 ячейкам (канал × продукт); при --out "
        "дублировать в sibling <out>.percell.csv",
    )
    bt.set_defaults(func=_cmd_backtest)


def _add_cost(sub: argparse._SubParsersAction[Any]) -> None:
    cs = sub.add_parser(
        "cost",
        help="полная себестоимость и contribution margin по 9 ячейкам",
    )
    cs.add_argument(
        "--current",
        required=True,
        help="текущий отчёт (.xls/.xlsx) компании 1..8",
    )
    cs.add_argument(
        "--train",
        nargs="*",
        default=[],
        help="отчёты своей фирмы за прошлые кварталы (калибровка yield/цены)",
    )
    cs.add_argument(
        "--history",
        nargs="*",
        default=[],
        help="доп. отчёты своей фирмы для калибровки",
    )
    cs.add_argument(
        "--yield-halflife",
        type=float,
        default=YIELD_HALFLIFE,
        help=f"полураспад EWMA yield-факторов, кв. (дефолт {YIELD_HALFLIFE})",
    )
    cs.add_argument("--out", help="сохранить смету по ячейкам в CSV")
    cs.set_defaults(func=_cmd_cost)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gmc-forecaster",
        description="Прогноз и оценка модели спроса (GMC)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_forecast(sub)
    _add_backtest(sub)
    _add_cost(sub)
    return ap


def _sig(p: float) -> str:
    """Звёзды значимости по p-value."""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


# блоки коэффициентов для режимов --coeffs
_KEY_BLOCKS = ["const", "признак", "наклон"]
_FE_BLOCKS = ["ячейка", "группа", "фирма", "прочее"]


def _print_coeffs(df: pd.DataFrame, level: str) -> None:
    """Человекочитаемый блок коэффициентов модели доли с значимостью."""
    if level == "none":
        return
    cs = df.attrs.get("coef_summary")
    fit = df.attrs.get("fit", {})
    if fit.get("degenerate"):
        print()
        print(
            "Коэффициенты модели доли не выводятся: --train не содержит "
            "конкурентных данных с долями рынка (только история/клоны) — "
            "эффекты цены/рекламы/рейтинга не идентифицируются (подгонка "
            "вырождена: остатки схлопнуты, ст.ошибки ≈ 0, t огромны, p "
            "обнуляются). Добавьте в --train регулярные отчёты с долями "
            "(W…), чтобы оценить рычаг решений."
        )
        return
    if cs is None or cs.empty:
        return
    ridge = float(fit.get("ridge", 0.0))
    print()
    print(
        f"Коэффициенты модели доли (стадия 1) | R²={fit.get('r2')} "
        f"n={fit.get('n')} edf={fit.get('edf')} | ridge λ={ridge:.3g} "
        f"(усадка наклонов u_g к global; λ↑ = сильнее пулинг)"
    )
    blocks = _KEY_BLOCKS + (_FE_BLOCKS if level == "full" else [])
    view = cs[cs["блок"].isin(blocks)]
    print(f"  {'признак':<24}{'коэф':>8}{'ст.ош':>8}{'t':>7}{'p':>8}  знач")
    for _, r in view.iterrows():
        print(
            f"  {str(r['признак']):<24}{float(r['коэф']):>8.3f}"
            f"{float(r['ст.ош']):>8.3f}{float(r['t']):>7.2f}"
            f"{float(r['p']):>8.3f}  {_sig(float(r['p']))}"
        )
    bp = cs.loc[cs["col"] == "log_price", "коэф"]
    if not bp.empty:
        b = float(bp.iloc[0])
        s_ser = (
            df["доля_сцен_%"].dropna()
            if "доля_сцен_%" in df
            else pd.Series(dtype=float)
        )
        s = float(s_ser.median()) if not s_ser.empty else float("nan")
        if math.isnan(
            s
        ):  # доли не наблюдаемы (1-я итерация) — рычаг заимствован
            print(
                f"  Смысл: β_price={b:.2f} → полулог-эластичность спроса по "
                f"цене (доли не наблюдаемы — заимствованный рычаг)."
            )
        else:
            print(
                f"  Смысл: β_price={b:.2f} → эластичность доли по цене ≈ "
                f"β_price·(1−s); при s={s:.0f}% ≈ {b * (1 - s / 100):.2f}."
            )
    print(
        "  Значимость: *** p<0.01  ** p<0.05  * p<0.1 (нормальное "
        "приближение; наклоны u_g штрафуются → приблизит.)"
    )


def _cmd_forecast(a: argparse.Namespace) -> None:
    df = forecast(
        a.current,
        a.train,
        a.history,
        k_n=a.dist_k_n,
        k_comm=a.dist_k_comm,
        lever_k=a.lever_k,
    )
    m = df.attrs["meta"]
    print(
        f"Компания {m['company']}, группа {m['group']}: "
        f"прогноз Q{m['q_now']}->Q{m['q_next']} "
        f"(сезонный множитель объёма {m['seas_ratio']}, "
        f"демпфер рычага k={m['lever_k']})"
    )
    if m.get("mode") == "history":
        print(
            "⚠ 1-я итерация: своих долей рынка нет (history-рамп). Бейзлайн = "
            "персистенция наблюдаемого спроса; рычаг решений — ЗАИМСТВОВАННАЯ "
            "эластичность из --train (регулярные отчёты), низкая уверенность. "
            "Доля_сцен_% не оценивается; Δ_рычаг_% = эффект своих решений."
        )
    col = "Δ_дистриб_след_%"
    if col in df and (df[col].fillna(0) != 0).any():
        q_after = m["q_next"] % 4 + 1
        print(
            f"⚠ Дистрибьюторы: найм выходит на охват лишь со следующего "
            f"квартала (Q{q_after}) — на прогноз Q{m['q_next']} НЕ влияет. Его "
            f"отложенный эффект — в колонке {col}."
        )
    if "прибыль_ячейка" not in df.columns:
        print(
            "⚠ Себестоимость/прибыль не рассчитаны: нет производственных данных "
            "(листы затрат отчёта относятся к компании вне 1..8 / history-"
            "рамп). Колонки себест_полн_ед/CM_ед/прибыль_ячейка пропущены."
        )
    print(df.to_string(index=False))
    _print_coeffs(df, a.coeffs)
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"-> {a.out}", file=sys.stderr)
        cs = df.attrs.get("coef_summary")
        degenerate = bool(df.attrs.get("fit", {}).get("degenerate"))
        if (
            a.coeffs != "none"
            and not degenerate
            and cs is not None
            and not cs.empty
        ):
            cs = cs.copy()
            cs["знач"] = cs["p"].map(_sig)
            stem, ext = os.path.splitext(a.out)
            cpath = f"{stem}.coef{ext or '.csv'}"
            cs.to_csv(cpath, index=False)
            print(f"-> {cpath}", file=sys.stderr)


def _fmt(x: float | None, suffix: str = "") -> str:
    return "н/д" if x is None else f"{x:.2f}{suffix}"


def _print_per_cell(pc: pd.DataFrame) -> None:
    """9-строчная таблица метрик по ячейкам (канал × продукт)."""
    if pc.empty:
        return
    print()
    print("По ячейкам (канал × продукт) | доля-MAE п.п., объём/спрос MAPE %:")
    print(
        f"  {'канал':<6}{'пр':>3}{'n':>3}{'дол.real':>9}{'дол.orc':>9}"
        f"{'объём':>8}{'спр.real':>9}{'спр.orc':>9}{'персист':>9}"
        f"{'сез.наив':>9}"
    )
    for _, r in pc.iterrows():
        print(
            f"  {str(r['канал']):<6}{int(r['продукт']):>3}{int(r['n']):>3}"
            f"{_fmt(_fin(r['доля_MAE_real'])):>9}"
            f"{_fmt(_fin(r['доля_MAE_oracle'])):>9}"
            f"{_fmt(_fin(r['объём_MAPE'])):>8}"
            f"{_fmt(_fin(r['спрос_MAPE_real'])):>9}"
            f"{_fmt(_fin(r['спрос_MAPE_oracle'])):>9}"
            f"{_fmt(_fin(r['спрос_MAPE_persist'])):>9}"
            f"{_fmt(_fin(r['спрос_MAPE_seasnaive'])):>9}"
        )


def _cmd_backtest(a: argparse.Namespace) -> None:
    summary, df = backtest(
        a.reports, a.train or None, a.history or None, lever_k=a.lever_k
    )
    s = summary
    print(
        f"Бэктест: {s['n_переходов']} переходов, {s['n_ячеек']} ячеек | "
        f"группы {s['группы']}, компании {s['компании']} | "
        f"сезонность: {'да' if s['сезонность'] else 'нет'}"
    )
    print("Стадия 1 — доля (MAE, п.п.):")
    print(f"  своя, realistic:    {_fmt(s['доля_MAE_своя_real'])}")
    print(f"  своя, oracle:       {_fmt(s['доля_MAE_своя_oracle'])}")
    print(f"  все 8 комп, oracle: {_fmt(s['доля_MAE_все_oracle'])}")
    print(f"Стадия 2 — объём (MAPE): {_fmt(s['объём_MAPE'], '%')}")
    print("Сквозной спрос (MAPE):")
    print(f"  realistic: {_fmt(s['спрос_MAPE_real'], '%')}")
    print(f"  oracle:    {_fmt(s['спрос_MAPE_oracle'], '%')}")
    print("Бейзлайны спроса (MAPE):")
    print(f"  персистенция:   {_fmt(s['спрос_MAPE_persist'], '%')}")
    print(f"  сезонный наив:  {_fmt(s['спрос_MAPE_seasnaive'], '%')}")
    pc = per_cell_summary(df) if a.per_cell else None
    if pc is not None:
        _print_per_cell(pc)
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"-> {a.out}", file=sys.stderr)
        if pc is not None and not pc.empty:
            stem, ext = os.path.splitext(a.out)
            ppath = f"{stem}.percell{ext or '.csv'}"
            pc.to_csv(ppath, index=False)
            print(f"-> {ppath}", file=sys.stderr)


_COST_COLS = [
    "канал",
    "продукт",
    "ед",
    "цена",
    "выручка",
    "материал",
    "конверсия",
    "дистриб",
    "накладные",
    "себест_полн_ед",
    "CM_ед",
    "прибыль_полн_ед",
]


def _cmd_cost(a: argparse.Namespace) -> None:
    try:
        d = cost(
            a.current, a.train, a.history, yield_halflife=a.yield_halflife
        )
    except ValueError as e:
        print(f"⚠ {e}", file=sys.stderr)
        sys.exit(1)
    m = d.attrs["meta"]
    y = d.attrs["yields"]
    pl = d.attrs["pl_check"]
    print(
        f"Компания {m['company']}, группа {m['group']}: себестоимость "
        f"{m['year']}Q{m['quarter']} | смен {m['shifts']} | спот-цена сырья "
        f"{m['mat_price']} ерз/шт"
    )
    print(
        f"yield (EWMA): брак {y['defect_rate'] * 100:.1f}%  "
        f"болезни {y['absence_rate'] * 100:.1f}%  "
        f"эфф.станков {y['efficiency'] * 100:.1f}%"
    )
    print(d[_COST_COLS].round(1).to_string(index=False))
    print(
        f"Сверка: смета {pl['смета']:.0f} vs факт COGS+накл "
        f"{pl['факт_cogs_плюс_накл']:.0f} (разрыв — база материала «продано» "
        f"vs «использовано»); машино-часы расч {pl['машино_часы_расч']:.0f} "
        f"vs факт {pl['машино_часы_факт']:.0f}"
    )
    if a.out:
        d.to_csv(a.out, index=False)
        print(f"-> {a.out}", file=sys.stderr)


def main() -> None:
    a = build_parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
