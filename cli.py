#!/usr/bin/env python3
"""
gmc_cli.py — командный интерфейс к парсеру GMC-отчётов.

Примеры:
  python gmc_cli.py W115264.xls                 # таблица в stdout
  python gmc_cli.py W115264.xls --json out.json
  python gmc_cli.py W115264.xls --csv  out.csv
  python gmc_cli.py W115264.xls --explain       # текстовый бриф для агента
  python gmc_cli.py data/*.xls --history hist.csv   # склеить историю (X_t, y_{t+1})
"""

import argparse
import json
import sys
import pandas as pd
from parser_flat import (
    parse_report_flat as parse_report,
)

RU = {"EAEU": "ЕАЭС", "ASEAN": "АСЕАН", "INT": "Интернет"}

type MetaDict = dict[str, object]


def explain(meta: MetaDict, df: pd.DataFrame) -> str:
    """Компактный человекочитаемый бриф одного квартала (для LLM-агента)."""
    L = [
        f"Отчёт GMC: компания {meta['company']}, группа {meta['group']}, "
        f"{meta['year']} г., квартал {meta['quarter']}.",
        f"Макро: ВВП ЕАЭС={meta['gdp_eaeu']}, АСЕАН={meta['gdp_asean']}, "
        f"курс ерз/$={meta['fx_erz_usd']}.",
        f"Интернет-канал: посещений={meta['inet_visits']:.0f}, "
        f"%незашедших={meta['inet_noenter']}, жалоб={meta['inet_complaints']:.0f}.",
        "Спрос (ЗАКАЗЫ) по каналам×продуктам с ключевыми рычагами:",
    ]
    for _, r in df.iterrows():
        L.append(
            f"  {RU[r['channel']]}·P{r['product']}: спрос={r['demand']:.0f}, "
            f"цена={r['price']:.0f} (отн.конкур.={r['price_rel']:.2f}), "
            f"реклама={r['ad_product']:.0f}, дистриб.={r['dist_n']:.0f}@{r['dist_comm']:.0f}%, "
            f"доля={r['share_own']}, разработка={r['new_dev']}."
        )
    L.append(
        "Важно: спрос(ЗАКАЗЫ) ≠ продажи(ограничены отгрузкой); "
        "доля рынка считается по продажам, не по спросу."
    )
    return "\n".join(L)


def load_history(paths: list[str]) -> pd.DataFrame:
    """
    Склеить несколько отчётов и построить обучающую таблицу переходов:
    признаки квартала t + target = спрос квартала t+1 (demand_next)
    по каждой паре (channel, product). Сортировка по (year,quarter).
    """
    frames = []
    for p in paths:
        _, df = parse_report(p)
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    all_df["t"] = all_df["year"] * 4 + (
        all_df["quarter"] - 1
    )  # абсолютный номер кв.
    all_df = all_df.sort_values(["channel", "product", "t"])
    # target = спрос СЛЕДУЮЩЕГО квартала той же ячейки
    all_df["demand_next"] = all_df.groupby(["channel", "product"])[
        "demand"
    ].shift(-1)
    # оставляем только строки, где есть t+1
    return all_df.dropna(subset=["demand_next"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Парсер отчётов Global Management Challenge"
    )
    ap.add_argument("files", nargs="+", help="путь(и) к .xls отчёту(ам)")
    ap.add_argument("--json", metavar="OUT", help="выгрузить record в JSON")
    ap.add_argument(
        "--csv", metavar="OUT", help="выгрузить длинную таблицу в CSV"
    )
    ap.add_argument(
        "--explain", action="store_true", help="текстовый бриф для агента"
    )
    ap.add_argument(
        "--history",
        metavar="OUT",
        help="склеить все files в таблицу переходов (X_t -> demand_next)",
    )
    a = ap.parse_args()

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


if __name__ == "__main__":
    main()
