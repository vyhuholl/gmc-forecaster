"""
cli.py — точка входа `gmc-forecaster`. Подкоманды:
  • forecast — прогноз спроса на следующий квартал под сценарием решений;
  • backtest — оценка качества модели на исторических данных (без сценария).
"""

from __future__ import annotations
import argparse
import json
import sys
from typing import Any
from .forecast import forecast
from .backtest import backtest


def _add_forecast(sub: argparse._SubParsersAction[Any]) -> None:
    fc = sub.add_parser(
        "forecast", help="прогноз спроса под сценарием решений"
    )
    fc.add_argument(
        "--current", required=True, help="текущий отчёт (.xls/.xlsx)"
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
    fc.add_argument("--scenario", required=True, help="scenario.json")
    fc.add_argument("--out", help="сохранить прогноз в CSV")
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
    bt.add_argument("--out", help="сохранить детализацию по ячейкам в CSV")
    bt.set_defaults(func=_cmd_backtest)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gmc-forecaster",
        description="Прогноз и оценка модели спроса (GMC)",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_forecast(sub)
    _add_backtest(sub)
    return ap


def _cmd_forecast(a: argparse.Namespace) -> None:
    scenario = json.load(open(a.scenario, encoding="utf-8"))
    df = forecast(a.current, a.train, a.history, scenario)
    m = df.attrs["meta"]
    print(
        f"Компания {m['company']}, группа {m['group']}: "
        f"прогноз Q{m['q_now']}->Q{m['q_next']} "
        f"(сезонный множитель объёма {m['seas_ratio']})"
    )
    print(df.to_string(index=False))
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"-> {a.out}", file=sys.stderr)


def _fmt(x: float | None, suffix: str = "") -> str:
    return "н/д" if x is None else f"{x:.2f}{suffix}"


def _cmd_backtest(a: argparse.Namespace) -> None:
    summary, df = backtest(a.reports, a.train or None, a.history or None)
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
    if a.out:
        df.to_csv(a.out, index=False)
        print(f"-> {a.out}", file=sys.stderr)


def main() -> None:
    a = build_parser().parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
