"""
report.py — человекочитаемый бриф по одному кварталу и склейка истории
переходов (X_t -> demand_next) для команды `parse`.
"""

from __future__ import annotations
import pandas as pd
from .parser_flat import parse_report_flat as parse_report

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
