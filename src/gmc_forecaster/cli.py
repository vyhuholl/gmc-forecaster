"""
cli.py — единая точка входа `gmc-forecaster` с подкомандами:
  parse     — разбор отчёта(ов) GMC (лист 'W');
  forecast  — прогноз спроса под сценарием решений;
  backtest  — бэктест прозрачной модели спроса на истории.
"""

from __future__ import annotations
import argparse
import json
import sys
from .report import explain, load_history
from .parser_flat import parse_report_flat as parse_report
from .forecast import forecast
from .features import build_features
from .model import backtest


def cmd_parse(a: argparse.Namespace) -> None:
    if a.history:
        hist = load_history(a.files)
        hist.to_csv(a.history, index=False)
        print(
            f"История: {len(hist)} строк переходов -> {a.history}",
            file=sys.stderr,
        )
        print(
            hist[
                [
                    "channel",
                    "product",
                    "year",
                    "quarter",
                    "demand",
                    "demand_next",
                ]
            ].to_string(index=False)
        )
        return

    meta, df = parse_report(a.files[0])
    if a.explain:
        print(explain(meta, df))
        return
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(
                {"meta": meta, "cells": df.to_dict("records")},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"-> {a.json}", file=sys.stderr)
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"-> {a.csv}", file=sys.stderr)
    if not (a.json or a.csv):
        print(df.to_string(index=False))


def cmd_forecast(a: argparse.Namespace) -> None:
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


def cmd_backtest(a: argparse.Namespace) -> None:
    df = build_features(a.files)
    res, detail = backtest(df, holdout_quarter=a.holdout)
    for k, v in res.items():
        print(f"{k}: {v}")
    print("\nпо ячейкам (truth vs pred):")
    print(
        detail[["channel", "product", "lag1", "truth", "pred"]].to_string(
            index=False
        )
    )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gmc-forecaster",
        description="Прогнозирование спроса в Global Management Challenge",
    )
    sub = ap.add_subparsers(dest="command", required=True)

    p_parse = sub.add_parser(
        "parse", help="разобрать отчёт(ы) GMC (.xls/.xlsx)"
    )
    p_parse.add_argument("files", nargs="+", help="путь(и) к .xls отчёту(ам)")
    p_parse.add_argument(
        "--json", metavar="OUT", help="выгрузить record в JSON"
    )
    p_parse.add_argument(
        "--csv", metavar="OUT", help="выгрузить длинную таблицу в CSV"
    )
    p_parse.add_argument(
        "--explain", action="store_true", help="текстовый бриф для агента"
    )
    p_parse.add_argument(
        "--history",
        metavar="OUT",
        help="склеить все files в таблицу переходов (X_t -> demand_next)",
    )
    p_parse.set_defaults(func=cmd_parse)

    p_fc = sub.add_parser(
        "forecast", help="прогноз спроса под сценарием решений"
    )
    p_fc.add_argument(
        "--current", required=True, help="текущий отчёт (.xls/.xlsx)"
    )
    p_fc.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="конкурентные отчёты для обучения",
    )
    p_fc.add_argument(
        "--history",
        nargs="*",
        default=[],
        help="файлы группы 0 для сезонности",
    )
    p_fc.add_argument("--scenario", required=True, help="scenario.json")
    p_fc.add_argument("--out", help="сохранить прогноз в CSV")
    p_fc.set_defaults(func=cmd_forecast)

    p_bt = sub.add_parser(
        "backtest", help="бэктест модели спроса на истории отчётов"
    )
    p_bt.add_argument("files", nargs="+", help="история отчётов (.xls/.xlsx)")
    p_bt.add_argument(
        "--holdout",
        type=int,
        default=3,
        metavar="Q",
        help="квартал для валидации (по умолчанию 3)",
    )
    p_bt.set_defaults(func=cmd_backtest)
    return ap


def main() -> None:
    ap = build_parser()
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
