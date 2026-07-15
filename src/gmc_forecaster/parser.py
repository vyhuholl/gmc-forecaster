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
from dataclasses import dataclass, field
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

# --- позиционная схема ПЕРВОГО листа 'Your decisions' (0-based row, col) ---
# Выведена и провалидирована по 4 текущим отчётам (разные группы/компании);
# как и SCHEMA листа 'W', позиции стабильны и не зависят от языка формы.
# Строки рекламы/дистрибьюторов и цен — по каналам; столбцы — по продуктам.
DEC_AD_ROW = {"EAEU": 13, "ASEAN": 14, "INT": 15}  # реклама + дистрибьюторы
DEC_PRICE_ROW = {"EAEU": 18, "ASEAN": 19, "INT": 20}
DEC_PLAN_ROW = {"EAEU": 23, "ASEAN": 24, "INT": 25}  # план поставок по каналам
DEC_PROD_COL = {1: 5, 2: 7, 3: 9}  # столбцы продуктов 1/2/3
DEC_IMAGE_COL = 4  # имидж-реклама канала
DEC_DIST_N_COL = 15  # число агентов/дистрибьюторов канала
DEC_DIST_COMM_COL = 22  # комиссия дистрибьюторов, %
DEC_ASM_ROW = 30  # время сборки по продуктам (мин), столбцы DEC_PROD_COL
DEC_SHIFTS = (19, 22)  # число рабочих смен (row, col)
DEC_ASM_HIRE = (23, 15)  # сборщики: найм(+)/увольнение(−)
DEC_ASM_WAGE = (24, 15)  # часовая ставка сборщиков, ерз
DEC_MACH_BUY = (30, 15)  # покупка станков, шт
DEC_MACH_SELL = (30, 22)  # продажа станков, шт


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


def _decnum(x: Any) -> float | None:
    """числовое значение ячейки листа решений или None (пусто/текст)."""
    v = _num(x)
    return float(v) if isinstance(v, (int, float)) else None


def parse_decisions(path: str) -> dict[str, Any]:
    """Решения игрока с ПЕРВОГО листа ('Your decisions') отчёта по позиционной
    схеме DEC_SCHEMA (языконезависимо). Это заменяет scenario.json: игрок правит
    свои what-if цены/рекламу/дистрибьюторов прямо в Excel. Возвращает dict:
      price{ячейка}            — абс. цены (ерз) по 9 ячейкам,
      plan{ячейка}             — план поставок (ед) по 9 ячейкам (= planned
                                 отчёта; оборотная база себестоимости),
      assembly_min{продукт}    — фактическое время сборки (мин/ед) по 3 продуктам
                                 (для этой игры; сверено с assembler_worked_h),
      adspend_total            — суммарный бюджет рекламы (тыс. ерз): имидж по
                                 3 каналам + товарная по 9 ячейкам (= уровень
                                 компании модели W[adspend_base]/1000),
      dist_n{канал}            — число дистрибьюторов,
      dist_comm{канал}         — комиссия дистрибьюторов, %,
      shifts                   — число рабочих смен (или None),
      hire                     — найм(+)/увольнение(−) сборщиков (или None),
      wage                     — часовая ставка сборщиков, ерз (или None),
      mach_buy / mach_sell     — покупка/продажа станков, шт (или None).
    Пустая ячейка -> ключ опускается/None (наследует текущее из --current)."""
    df = pd.read_excel(path, sheet_name=0, header=None, engine="calamine")

    def at(r: int, c: int) -> float | None:
        try:
            return _decnum(df.iat[r, c])
        except IndexError:
            return None

    price: dict[str, float] = {}
    plan: dict[str, float] = {}
    assembly_min: dict[int, float] = {}
    dist_n: dict[str, float] = {}
    dist_comm: dict[str, float] = {}
    ad_total = 0.0
    for ch in CHANNELS:
        for p in PRODUCTS:
            v = at(DEC_PRICE_ROW[ch], DEC_PROD_COL[p])
            if v is not None:
                price[f"{ch}{p}"] = v
            pl = at(DEC_PLAN_ROW[ch], DEC_PROD_COL[p])
            if pl is not None:
                plan[f"{ch}{p}"] = pl
        ar = DEC_AD_ROW[ch]
        img = at(ar, DEC_IMAGE_COL)
        if img is not None:
            ad_total += img
        for p in PRODUCTS:
            v = at(ar, DEC_PROD_COL[p])
            if v is not None:
                ad_total += v
        n = at(ar, DEC_DIST_N_COL)
        if n is not None:
            dist_n[ch] = n
        comm = at(ar, DEC_DIST_COMM_COL)
        if comm is not None:
            dist_comm[ch] = comm
    for p in PRODUCTS:
        a = at(DEC_ASM_ROW, DEC_PROD_COL[p])
        if a is not None:
            assembly_min[p] = a
    return {
        "price": price,
        "plan": plan,
        "assembly_min": assembly_min,
        "adspend_total": ad_total,
        "dist_n": dist_n,
        "dist_comm": dist_comm,
        "shifts": at(*DEC_SHIFTS),
        "hire": at(*DEC_ASM_HIRE),
        "wage": at(*DEC_ASM_WAGE),
        "mach_buy": at(*DEC_MACH_BUY),
        "mach_sell": at(*DEC_MACH_SELL),
    }


def parse_report(path: str) -> tuple[dict[str, Any], pd.DataFrame]:
    # engine='calamine' читает и старый BIFF .xls, и strict-OOXML .xlsx
    # (историю), где openpyxl не видит листов
    col = (
        pd.read_excel(path, sheet_name="W", header=None, engine="calamine")
        .iloc[:, 0]
        .tolist()
    )
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


# --- позиционные схемы листов затрат (0-based (row, col)) ---
# Выведены и провалидированы по текущим отчётам W-серии (RU-2026). Как и SCHEMA
# листа 'W', позиции стабильны и не зависят от языка формы. Продукты 1/2/3 лежат
# в столбцах 20/22/24 листа 'Resources and products'.
RES_PROD_COL = {1: 20, 2: 22, 3: 24}
RES_ROW = {  # строки триплетов «продукт 1/2/3»
    "planned": 5,  # Запланировано
    "produced": 6,  # Произведено
    "defects": 7,  # Брак
    "components": 41,  # Использовано в сборке (полуфабрикаты)
}
RES_SCALAR = {  # одиночные значения (row, col)
    "assembler_avail_h": (15, 14),  # доступно часов работы сборщиков
    "assembler_worked_h": (17, 14),  # фактически отработано часов
    "assembler_absence_h": (16, 14),  # общее время отсутствия/болезни
    "assemblers": (6, 13),  # сборщиков на начало прош. кв.
    "mechanics": (6, 14),  # механиков на начало прош. кв.
    "machines_used": (18, 6),  # станков использовалось
    "machines_next": (20, 6),  # станков доступно для след. кв.
    "machine_avail_h": (22, 6),  # доступных станко-часов
    "machine_worked_h": (24, 6),  # отработано станко-часов
    "machine_downtime_h": (23, 6),  # часов простоя станков
    "machine_efficiency": (26, 6),  # средняя эффективность станка, %
    "material_used": (33, 6),  # использовано сырья, шт
    "material_purchased_units": (30, 6),  # закуплено сырья, шт
}
# лист 'Financial statements': накладные — метки в столбце 2, значения в 5;
# P&L (себестоимость производства) — метки в 8, значения в 11.
FIN_OVERHEAD_COL = 5
FIN_OVERHEAD = {  # {ключ: row} — накладные расходы
    "реклама": 7,
    "интернет_агент": 8,
    "интернет_провайдер": 9,
    "агенты_дистриб": 10,
    "офис_продаж": 11,
    "гарантия": 12,
    "rnd": 13,
    "веб": 14,
    "персонал": 15,
    "техобсл": 16,
    "склад_закупка": 17,
    "маркет_исслед": 18,
    "кредит_контроль": 19,
    "страховка": 20,
    "упр_бюджет": 21,
    "прочие": 22,
}
FIN_OVERHEAD_TOTAL_ROW = 23  # Итого накладные расходы
FIN_PL_COL = 11
FIN_PL = {  # {ключ: row} — отчёт о прибылях/убытках (производство)
    "revenue": 7,  # выручка от реализации
    "components_purchased": 10,  # закупленные полуфабрикаты
    "material_purchased": 11,  # закупленное сырьё (деньги)
    "machine_opex": 12,  # затраты на эксплуатацию станков
    "mechanic_salary": 13,  # зарплаты механиков
    "assembler_salary": 14,  # зарплаты сборщиков
    "qc": 15,  # контроль качества (ОТК)
    "transport": 16,  # аренда транспорта
    "cogs": 18,  # затраты на пр-во и реализацию продукции
    "gross_profit": 19,  # валовая прибыль/убыток
    "depreciation": 22,  # амортизация
}


@dataclass(frozen=True)
class CostReport:
    """Производственные и стоимостные данные одного отчёта (листы
    'Resources and products' и 'Financial statements'). Всё в ерз, если не
    указано иное; часы — станко-/человеко-часы прошлого квартала."""

    group: int
    company: int
    year: int
    quarter: int
    # производство по продуктам (ключи 1/2/3)
    produced: dict[int, float]
    defects: dict[int, float]
    components_used: dict[int, float]  # собрано из полуфабрикатов
    planned: dict[int, float]
    # сборщики / механики
    assembler_avail_h: float
    assembler_worked_h: float
    assembler_absence_h: float
    assemblers: float
    mechanics: float
    # станки
    machines: float
    machine_avail_h: float
    machine_worked_h: float
    machine_downtime_h: float
    machine_efficiency: float  # %
    # материалы (в штуках)
    material_used: float
    material_purchased_units: float
    # финансы: P&L производства
    revenue: float
    components_purchased_cost: float
    material_purchased_cost: float
    machine_opex: float
    mechanic_salary: float
    assembler_salary: float
    qc_cost: float
    transport_cost: float
    cogs: float
    depreciation: float
    # накладные (именованные статьи + итог)
    overhead: dict[str, float] = field(default_factory=dict)
    overhead_total: float = 0.0


def parse_costs(path: str) -> CostReport:
    """Данные затрат отчёта по позиционным схемам RES_SCHEMA/FIN_SCHEMA
    (языконезависимо, движок calamine). Пустая/текстовая ячейка -> 0.0
    (эффективность -> 100.0). Валидно для отчёта компании 1..8 W-серии."""
    res = pd.read_excel(
        path,
        sheet_name="Resources and products",
        header=None,
        engine="calamine",
    )
    fin = pd.read_excel(
        path,
        sheet_name="Financial statements",
        header=None,
        engine="calamine",
    )
    meta, _ = parse_report(path)

    def at(df: pd.DataFrame, r: int, c: int, default: float = 0.0) -> float:
        try:
            v = _num(df.iat[r, c])
        except IndexError:
            return default
        return float(v) if isinstance(v, (int, float)) else default

    def triple(row: int) -> dict[int, float]:
        return {p: at(res, row, RES_PROD_COL[p]) for p in PRODUCTS}

    overhead = {
        k: at(fin, row, FIN_OVERHEAD_COL) for k, row in FIN_OVERHEAD.items()
    }
    return CostReport(
        group=meta["group"],
        company=meta["company"],
        year=meta["year"],
        quarter=meta["quarter"],
        produced=triple(RES_ROW["produced"]),
        defects=triple(RES_ROW["defects"]),
        components_used=triple(RES_ROW["components"]),
        planned=triple(RES_ROW["planned"]),
        assembler_avail_h=at(res, *RES_SCALAR["assembler_avail_h"]),
        assembler_worked_h=at(res, *RES_SCALAR["assembler_worked_h"]),
        assembler_absence_h=at(res, *RES_SCALAR["assembler_absence_h"]),
        assemblers=at(res, *RES_SCALAR["assemblers"]),
        mechanics=at(res, *RES_SCALAR["mechanics"]),
        machines=at(res, *RES_SCALAR["machines_used"]),
        machine_avail_h=at(res, *RES_SCALAR["machine_avail_h"]),
        machine_worked_h=at(res, *RES_SCALAR["machine_worked_h"]),
        machine_downtime_h=at(res, *RES_SCALAR["machine_downtime_h"]),
        machine_efficiency=at(
            res, *RES_SCALAR["machine_efficiency"], default=100.0
        ),
        material_used=at(res, *RES_SCALAR["material_used"]),
        material_purchased_units=at(
            res, *RES_SCALAR["material_purchased_units"]
        ),
        revenue=at(fin, FIN_PL["revenue"], FIN_PL_COL),
        components_purchased_cost=at(
            fin, FIN_PL["components_purchased"], FIN_PL_COL
        ),
        material_purchased_cost=at(
            fin, FIN_PL["material_purchased"], FIN_PL_COL
        ),
        machine_opex=at(fin, FIN_PL["machine_opex"], FIN_PL_COL),
        mechanic_salary=at(fin, FIN_PL["mechanic_salary"], FIN_PL_COL),
        assembler_salary=at(fin, FIN_PL["assembler_salary"], FIN_PL_COL),
        qc_cost=at(fin, FIN_PL["qc"], FIN_PL_COL),
        transport_cost=at(fin, FIN_PL["transport"], FIN_PL_COL),
        cogs=at(fin, FIN_PL["cogs"], FIN_PL_COL),
        depreciation=at(fin, FIN_PL["depreciation"], FIN_PL_COL),
        overhead=overhead,
        overhead_total=at(fin, FIN_OVERHEAD_TOTAL_ROW, FIN_OVERHEAD_COL),
    )
