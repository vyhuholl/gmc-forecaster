"""
Разбор отчёта GMC по ЕДИНСТВЕННОМУ последнему листу 'W'
(плоский числовой экспорт). Читает только 'W', не трогает именованные листы,
поэтому полностью независим от языка содержимого и имён листов.

Позиционная схема выведена и провалидирована по 9 файлам (RU-2026, польский-2017,
испанский-2018, история-2024). Индексы 0-based в столбце листа 'W'. Порядок ячеек
внутри блоков: product-major, channel-minor -> off = 3*(product-1) + ch,
где ch: EAEU=0, ASEAN=1, INT=2.

Что НЕ экспортируется в 'W' (в отличие от именованных листов): процентные ставки,
курс валюты, % незашедших интернет-пользователей -> в meta вернутся None.
"""

from __future__ import annotations
from typing import Any
import pandas as pd

CHANNELS = ["EAEU", "ASEAN", "INT"]
PRODUCTS = [1, 2, 3]
CH = {"EAEU": 0, "ASEAN": 1, "INT": 2}

# --- позиционная схема листа 'W' (0-based индексы) ---
SCHEMA = {
    "group": 0,
    "company": 1,
    "year": 3,
    "quarter": 4,
    "ad_image_base": 6,  # 6 + ch
    "ad_product_base": 10,  # 10 + 3*(p-1) + ch
    "dist_base": 60,  # per channel triple (n, reward, comm) @ 60+3*ch (+0/+1/+2)
    "shipped_base": 120,  # 120 + 3*(p-1) + ch
    "demand_base": 130,
    "sold_base": 140,
    "stock_base": 160,
    "new_dev_base": 176,  # 176 + (p-1)  (токены Major/Minor)
    "visits": 318,
    "complaints": 328,
    "share_base": 331,  # 331 + 10*(co-1) + 3*(p-1) + ch
    "adspend_base": 421,  # 421 + 7*(co-1)
    "rating_base": 423,  # 423 + 7*(co-1) + (p-1);  web = 426 + 7*(co-1)
    "web_rating_off": 3,  # смещение website-рейтинга от rating_base компании
    "gdp_row": 503,
    "gdp_eaeu": 504,
    "gdp_asean": 505,
    "price_base": 525,  # 525 + 20*(co-1) + 3*(p-1) + ch
    "share_co_stride": 10,
    "rating_co_stride": 7,
    "price_co_stride": 20,
}


def _num(x: Any) -> Any:
    """нормализация ячейки 'W': число / текст / пусто(None)."""
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        try:
            return float(s)
        except ValueError:
            return s
    return None if pd.isna(x) else x


def _stars(x: Any) -> int | None:
    return x.count("*") if isinstance(x, str) and "*" in x else None


def _off(p: int, ch: str) -> int:
    return 3 * (p - 1) + CH[ch]


def parse_report(path: str) -> tuple[dict[str, Any], pd.DataFrame]:
    col = pd.read_excel(path, sheet_name="W", header=None).iloc[:, 0].tolist()
    W = [_num(v) for v in col]
    S = SCHEMA

    company = W[S["company"]]
    company = int(company) if company is not None else 0
    valid_own = 1 <= company <= 8  # в истории company=0 -> own-полей нет

    meta = {
        "file": path,
        "year": int(W[S["year"]]),
        "quarter": int(W[S["quarter"]]),
        "company": company,
        "group": int(W[S["group"]]) if W[S["group"]] is not None else 0,
        "gdp_eaeu": W[S["gdp_eaeu"]],
        "gdp_asean": W[S["gdp_asean"]],
        "gdp_row": W[S["gdp_row"]],
        "rate_eaeu": None,
        "rate_asean": None,
        "fx_erz_usd": None,  # нет в 'W'
        "inet_visits": W[S["visits"]],
        "inet_noenter": None,  # noenter нет в 'W'
        "inet_complaints": W[S["complaints"]],
        "web_rating_own": _stars(
            W[
                S["rating_base"]
                + S["rating_co_stride"] * (company - 1)
                + S["web_rating_off"]
            ]
        )
        if valid_own
        else None,
    }

    def price(co: int, p: int, ch: str) -> Any:
        return W[
            S["price_base"] + S["price_co_stride"] * (co - 1) + _off(p, ch)
        ]

    def share(co: int, p: int, ch: str) -> Any:
        return W[
            S["share_base"] + S["share_co_stride"] * (co - 1) + _off(p, ch)
        ]

    def rating(co: int, p: int) -> int | None:
        return _stars(
            W[S["rating_base"] + S["rating_co_stride"] * (co - 1) + (p - 1)]
        )

    others = [c for c in range(1, 9) if c != company] if valid_own else []

    rows = []
    for ch in CHANNELS:
        for p in PRODUCTS:
            off = _off(p, ch)
            own_price = price(company, p, ch) if valid_own else None
            comp_prices = [
                price(c, p, ch) for c in others if price(c, p, ch) is not None
            ]
            comp_shares = [
                share(c, p, ch) for c in others if share(c, p, ch) is not None
            ]
            comp_ratings_raw = [rating(c, p) for c in others]
            comp_ratings = [x for x in comp_ratings_raw if x]
            rows.append(
                {
                    "channel": ch,
                    "product": p,
                    "demand": W[S["demand_base"] + off],
                    "sold": W[S["sold_base"] + off],
                    "shipped": W[S["shipped_base"] + off],
                    "stock": W[S["stock_base"] + off],
                    "price": own_price,
                    "ad_product": W[S["ad_product_base"] + off],
                    "ad_image": W[S["ad_image_base"] + CH[ch]],
                    "dist_n": W[S["dist_base"] + 3 * CH[ch]],
                    "dist_reward": W[S["dist_base"] + 3 * CH[ch] + 1],
                    "dist_comm": W[S["dist_base"] + 3 * CH[ch] + 2],
                    "price_comp_mean": (sum(comp_prices) / len(comp_prices))
                    if comp_prices
                    else None,
                    "price_comp_min": min(comp_prices)
                    if comp_prices
                    else None,
                    "price_rel": (
                        own_price / (sum(comp_prices) / len(comp_prices))
                    )
                    if comp_prices and own_price
                    else None,
                    "share_own": share(company, p, ch) if valid_own else None,
                    "share_comp_mean": (sum(comp_shares) / len(comp_shares))
                    if comp_shares
                    else None,
                    "rating_own": rating(company, p) if valid_own else None,
                    "rating_comp_mean": (sum(comp_ratings) / len(comp_ratings))
                    if comp_ratings
                    else None,
                    "new_dev": W[S["new_dev_base"] + (p - 1)],
                }
            )
    df = pd.DataFrame(rows)
    is_int = df["channel"] == "INT"
    df.loc[is_int, "inet_visits"] = meta["inet_visits"]
    df.loc[is_int, "inet_noenter"] = meta["inet_noenter"]
    df.loc[is_int, "inet_complaints"] = meta["inet_complaints"]
    for k in ("year", "quarter", "company", "group"):
        df[k] = meta[k]
    return meta, df
