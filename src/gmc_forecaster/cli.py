"""
cli.py — точка входа `gmc-forecaster`: прогноз спроса под сценарием решений.
"""

from __future__ import annotations
import argparse
import json
import sys
from .forecast import forecast


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gmc-forecaster",
        description="Прогноз спроса под сценарием решений (GMC)",
    )
    ap.add_argument(
        "--current", required=True, help="текущий отчёт (.xls/.xlsx)"
    )
    ap.add_argument(
        "--train",
        nargs="+",
        required=True,
        help="конкурентные отчёты для обучения",
    )
    ap.add_argument(
        "--history",
        nargs="*",
        default=[],
        help="файлы группы 0 для сезонности",
    )
    ap.add_argument("--scenario", required=True, help="scenario.json")
    ap.add_argument("--out", help="сохранить прогноз в CSV")
    return ap


def main() -> None:
    a = build_parser().parse_args()
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


if __name__ == "__main__":
    main()
