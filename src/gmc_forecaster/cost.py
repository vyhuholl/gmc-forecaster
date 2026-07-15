"""
cost.py — полная себестоимость и contribution margin решения по 9 ячейкам
(3 продукта × 3 канала) для GMC.

Модель себестоимости строится СНИЗУ ВВЕРХ из авторитетных констант справочника
GMC (Табл. 5–18) и рыночных/решенческих величин отчёта (спот-цена сырья, ставки
зарплат, комиссия), а не из строки COGS P&L (та смешивает движение запасов).

Ключевые тождества (сверены на данных до ~0.1 %):
  сырьё_использ = Σ_p (произв_p − п/ф_p) × {1,2,3}
  машино-часы   = Σ_p (произв_p − п/ф_p) × {60,75,120}/60 / эффективность
Полуфабрикат замещает СТАНКО-этап (изготовление деталей), но не сборку.

Разнесение на 9 ячеек — двухступенчатое (производство по ПРОДУКТУ, продажа по
КАНАЛУ):
  • материал (переменные) → в ячейку по единицам продукта;
  • конверсия (зарплаты сборщиков/механиков, эксплуатация станков, амортизация,
    ОТК) — производственный фикс-пул → по СБОРЩИКО-МИНУТАМ (ед_p × время_сборки_p);
  • канал-пул (дистрибуция, транспорт, интернет, гарантия) → к каналам (интернет —
    в INT; прочее — по выручке канала), внутри канала на продукты по единицам;
  • overhead (маркетинг + прочие непроизв. накладные) → ПО ВЫРУЧКЕ ячейки.

Три yield-поправки калибруются по истории своей фирмы (EWMA, не последний кв.):
брак → делить затраты на ГОДНЫЕ; болезни → фикс. зарплата на меньший выход
(конверсия/ед ↑); эффективность станков → больше станко-часов/ед.

Производственные фиксы пересобираются по СМЕНАМ (не берутся прошлым фактом):
смены = ⌈доступ_ст-часы/(станки×576)⌉ (Табл. 7), эксплуатация/техобслуживание/
механики — по формулам m3 (сверены на 4 кв.).

Два выхода на ячейку: полная себестоимость (absorption) и contribution margin
(цена − материал − полуфабрикат − комиссия дистрибьютора; конверсия — полу-фикс,
только в полную).
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from .parser import parse_report, parse_decisions, parse_costs, CostReport

RU = {"EAEU": "ЕАЭС", "ASEAN": "АСЕАН", "INT": "Интернет"}
CHANNELS = ["EAEU", "ASEAN", "INT"]
PRODUCTS = [1, 2, 3]

# ---- константы мануала GMC (Табл. 5–18; см. Приложение design.md) ----
RAW_PER_UNIT = {1: 1.0, 2: 2.0, 3: 3.0}  # шт сырья/ед (Табл. 5)
MACHINE_MIN = {1: 60.0, 2: 75.0, 3: 120.0}  # станко-мин/ед (Табл. 5)
ASSEMBLY_MIN = {1: 100.0, 2: 150.0, 3: 300.0}  # сборка мин/ед (Табл. 5)
MACHINE_H_PER_SHIFT = (
    576.0  # макс. станко-часов на станок за 1 смену (Табл. 7)
)
MACHINE_OPEX_PER_H = 8.0  # эксплуатация станка, ерз/машино-час (Табл. 10)
MACHINE_OVERHEAD_PER_UNIT = 3500.0  # накладные на 1 станок, ерз (Табл. 10)
SHIFT_CONTROL = 12500.0  # контроль за смену, ерз (Табл. 10)
SUPPLY_PLANNING_PER_UNIT = 1.0  # планирование поставок, ерз/ед (Табл. 10)
QC_PER_UNIT = 1.0  # ОТК, ерз/ед (Табл. 10)
MAINT_PER_H = 85.0  # техобслуживание, ерз/час (Табл. 6)
DEPRECIATION_RATE = 0.025  # амортизация 2.5 %/кв (Табл. 18)
MACHINE_COST = 300000.0  # стоимость станка, ерз (Табл. 18)
SHIFT_MECH_PREMIUM = {
    1: 0.0,
    2: 1 / 3,
    3: 2 / 3,
}  # премия механикам (Табл. 16)
MECH_PER_MACHINE = 4  # неквал. рабочих на станок за смену (Табл. 7)

# полураспад EWMA yield-факторов (кварталы): свежесть важнее долгого среднего
YIELD_HALFLIFE = 2.0

# статьи накладных, относимые НА КАНАЛ (не в overhead по выручке)
_INT_OVERHEAD = ("интернет_агент", "интернет_провайдер", "веб")
_CHANNEL_OVERHEAD = ("агенты_дистриб", "гарантия")


def _ewma(vals: list[float], halflife: float) -> float | None:
    """EWMA по списку (старые→новые); свежесть важнее. Пусто -> None, один
    элемент -> он сам (вырождение в точку)."""
    if not vals:
        return None
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    m = vals[0]
    for v in vals[1:]:
        m = alpha * v + (1.0 - alpha) * m
    return m


def raw_used(produced: dict[int, float], comp: dict[int, float]) -> float:
    """Расход сырья (шт): Σ_p (произв_p − п/ф_p) × норматив сырья."""
    return sum((produced[p] - comp[p]) * RAW_PER_UNIT[p] for p in PRODUCTS)


def machine_hours(
    produced: dict[int, float], comp: dict[int, float], efficiency_pct: float
) -> float:
    """Отработанные станко-часы: Σ_p (произв_p − п/ф_p) × станко-мин/60 /
    эффективность. Полуфабрикат замещает станко-этап."""
    eff = efficiency_pct / 100.0 if efficiency_pct > 0 else 1.0
    mins = sum((produced[p] - comp[p]) * MACHINE_MIN[p] for p in PRODUCTS)
    return mins / 60.0 / eff


def assembler_hours(produced: dict[int, float]) -> float:
    """Сборщико-часы: Σ_p произв_p × время_сборки/60 (п/ф сборку не замещают)."""
    return sum(produced[p] * ASSEMBLY_MIN[p] for p in PRODUCTS) / 60.0


def shifts_needed(mach_hours: float, machines: float) -> int:
    """Минимальное число смен, покрывающее требуемые станко-часы (Табл. 7)."""
    if machines <= 0:
        return 1
    return max(1, math.ceil(mach_hours / (machines * MACHINE_H_PER_SHIFT)))


def reconstruct_machine_opex(
    machines: float, shifts: int, mach_hours: float, total_units: float
) -> float:
    """Затраты на эксплуатацию станков (m3, ~99.8 %):
    станки×3500 + смены×12500 + станко-часы×8 + ед×планирование."""
    return (
        machines * MACHINE_OVERHEAD_PER_UNIT
        + shifts * SHIFT_CONTROL
        + mach_hours * MACHINE_OPEX_PER_H
        + total_units * SUPPLY_PLANNING_PER_UNIT
    )


def infer_raw_coeffs(reports: list[CostReport]) -> dict[int, float] | None:
    """SANITY-CHECK нормативов сырья: МНК-оценка расхода/ед по ≥3 кварталам.
    Возвращает {p: коэф} либо None (мало данных). Источник нормативов —
    константы; инференс лишь сверяет знак/масштаб (при подстановке п/ф МНК «в
    лоб» неустойчив, коэф. может уйти в минус)."""
    if len(reports) < 3:
        return None
    x = np.array(
        [
            [r.produced[p] - r.components_used[p] for p in PRODUCTS]
            for r in reports
        ]
    )
    y = np.array([r.material_used for r in reports])
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    return {p: float(coef[i]) for i, p in enumerate(PRODUCTS)}


# ---------- калибровка yield-факторов по истории своей фирмы ----------
def calibrate_yields(
    reports: list[CostReport], halflife: float = YIELD_HALFLIFE
) -> dict[str, float]:
    """Три yield-фактора (EWMA по кварталам своей фирмы, старые→новые):
      defect_rate    — брак/произведено;
      absence_rate   — время болезни/доступные часы сборщиков;
      efficiency     — эффективность станков (доля, 0..1).
    Множители: defect_infl=1/(1−брак), absence_infl=1/(1−болезнь)."""
    rs = sorted(reports, key=lambda r: r.year * 4 + r.quarter)
    defect, absence, eff = [], [], []
    for r in rs:
        prod = sum(r.produced.values())
        if prod > 0:
            defect.append(sum(r.defects.values()) / prod)
        if r.assembler_avail_h > 0:
            absence.append(r.assembler_absence_h / r.assembler_avail_h)
        if r.machine_efficiency > 0:
            eff.append(r.machine_efficiency / 100.0)
    defect_rate = _ewma(defect, halflife) or 0.0
    absence_rate = _ewma(absence, halflife) or 0.0
    efficiency = _ewma(eff, halflife) or 1.0
    return {
        "defect_rate": defect_rate,
        "absence_rate": absence_rate,
        "efficiency": efficiency,
        "defect_infl": 1.0 / (1.0 - defect_rate) if defect_rate < 1 else 1.0,
        "absence_infl": (
            1.0 / (1.0 - absence_rate) if absence_rate < 1 else 1.0
        ),
    }


def material_unit_price(
    reports: list[CostReport], current: CostReport
) -> float:
    """Спот-цена сырья (ерз/шт) = закупка_деньги/закупка_шт, сглаженная (медиана)
    по кварталам с ненулевой закупкой; фоллбэк — текущий отчёт, затем 0."""
    prices = [
        r.material_purchased_cost / r.material_purchased_units
        for r in reports
        if r.material_purchased_units > 0
    ]
    if prices:
        return float(np.median(prices))
    if current.material_purchased_units > 0:
        return (
            current.material_purchased_cost / current.material_purchased_units
        )
    return 0.0


# ---------- разнесение затрат на 9 ячеек ----------
def _cell_frame(
    current: str, plan: dict[str, float], prices: dict[str, float]
) -> pd.DataFrame:
    """Единицы и цена по 9 ячейкам из РЕШЕНИЙ игрока: объём = план поставок
    (plan{ячейка}), цена = решение (prices{ячейка}). Пустая ячейка решения →
    фоллбэк на факт листа 'W' (shipped/price) — наследование текущего."""
    _, own = parse_report(current)
    rows = []
    for _, r in own.iterrows():
        ch = str(r["channel"])
        p = int(r["product"])
        key = f"{ch}{p}"
        units = plan.get(key)
        if units is None:  # фоллбэк: фактически поставлено
            units = float(r["shipped"]) if pd.notna(r["shipped"]) else 0.0
        price = prices.get(key)
        if price is None:  # фоллбэк: цена листа 'W'
            price = float(r["price"]) if pd.notna(r["price"]) else 0.0
        rows.append(
            {
                "channel": ch,
                "product": p,
                "units": float(units),
                "price": float(price),
                "revenue": float(price) * float(units),
            }
        )
    return pd.DataFrame(rows)


def _allocate(
    df: pd.DataFrame,
    cur: CostReport,
    mat_price: float,
    yields: dict[str, float],
    dist_comm: dict[str, float],
    assembly_min: dict[int, float],
    ad_override: float | None,
    shifts_override: int | None,
    machines: float,
    demand: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Разносит все включённые затраты на 9 ячеек и возвращает (df, пулы).
    Объём/ячейка и цена — из решений (df уже собран `_cell_frame`); веса
    конверсии — по ФАКТИЧЕСКОМУ времени сборки решений (`assembly_min`, фоллбэк —
    константа мануала); число смен и парк станков — из решений (`shifts_override`,
    `machines`); реклама → маркетинговые накладные из решений (`ad_override`).
    Себест/ед и CM/ед считаются на базе ПЛАНА (производственный прогон) и НЕ
    зависят от базы продаж. При заданном `demand` (продано по ячейкам) добавляются
    реализационные колонки на базе прогноза (продано = min(demand, план))."""
    d = df.copy()
    n = len(d)
    units = d["units"].to_numpy(dtype=float)
    prod = d["product"].to_numpy(dtype=int)
    ch = d["channel"].to_numpy(dtype=object)
    revenue = d["revenue"].to_numpy(dtype=float)
    rev_tot = revenue.sum()

    # --- пул материала (переменные): по ячейке units×сырьё/ед×цена, ×брак ---
    raw_pu = np.array([RAW_PER_UNIT[int(p)] for p in prod])
    material = units * raw_pu * mat_price * yields["defect_infl"]

    # --- производственный фикс-пул (конверсия) по СБОРЩИКО-МИНУТАМ ---
    # состав: зарплаты сборщиков/механиков, эксплуатация станков (реконстр. по
    # сменам решения, volume-responsive), амортизация, ОТК; болезни ↑ труд.часть.
    # Веса — по фактическому времени сборки решений (константа лишь как фоллбэк).
    total_units = float(sum(cur.produced.values()))
    mh = machine_hours(
        cur.produced, cur.components_used, cur.machine_efficiency
    )
    shifts = (
        shifts_override
        if shifts_override is not None
        else shifts_needed(mh, machines)
    )
    machine_opex = reconstruct_machine_opex(machines, shifts, mh, total_units)
    depreciation = machines * MACHINE_COST * DEPRECIATION_RATE
    labor = (cur.assembler_salary + cur.mechanic_salary) * yields[
        "absence_infl"
    ]
    conv_pool = labor + machine_opex + depreciation + total_units * QC_PER_UNIT
    asm_pu = np.array(
        [assembly_min.get(int(p), ASSEMBLY_MIN[int(p)]) for p in prod]
    )
    asm_min = units * asm_pu
    w_conv = asm_min / asm_min.sum() if asm_min.sum() > 0 else np.zeros(n)
    conversion = conv_pool * w_conv

    # --- канал-пул: интернет → INT; дистрибуция/гарантия/транспорт → к каналу
    # по выручке; внутри канала на продукты по единицам ---
    channel = np.zeros(n)
    is_int = ch == "INT"
    int_pool = sum(cur.overhead.get(k, 0.0) for k in _INT_OVERHEAD)
    u_int = units[is_int].sum()
    if u_int > 0:
        channel[is_int] += int_pool * units[is_int] / u_int
    common_pool = (
        sum(cur.overhead.get(k, 0.0) for k in _CHANNEL_OVERHEAD)
        + cur.transport_cost
    )
    for c in CHANNELS:
        m = ch == c
        rev_c = revenue[m].sum()
        share_c = rev_c / rev_tot if rev_tot > 0 else 0.0
        pool_c = common_pool * share_c
        u_c = units[m].sum()
        if u_c > 0:
            channel[m] += pool_c * units[m] / u_c

    # --- overhead (маркетинг + прочие непроизв.) по ВЫРУЧКЕ ячейки; рекламу
    # заменяем бюджетом решений (ad_override), если задан ---
    used = set(_INT_OVERHEAD) | set(_CHANNEL_OVERHEAD)
    oh_pool = sum(v for k, v in cur.overhead.items() if k not in used)
    if ad_override is not None:
        oh_pool += ad_override - cur.overhead.get("реклама", 0.0)
    w_rev = revenue / rev_tot if rev_tot > 0 else np.zeros(n)
    overhead = oh_pool * w_rev

    full = material + conversion + channel + overhead
    price = d["price"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        full_pu = np.where(units > 0, full / units, 0.0)
    # contribution margin = цена − материал − п/ф − комиссия дистрибьютора
    comm_pu = np.array(
        [price[i] * dist_comm.get(str(ch[i]), 0.0) / 100.0 for i in range(n)]
    )
    mat_pu = raw_pu * mat_price  # номинальный материал/ед (без брак-инфляции)
    cm_pu = price - mat_pu - comm_pu

    d["материал"] = material
    d["конверсия"] = conversion
    d["дистриб"] = channel  # канал-пул (иначе коллизия с колонкой канала)
    d["накладные"] = overhead
    d["себест_полн"] = full
    d["себест_полн_ед"] = full_pu
    d["CM_ед"] = cm_pu
    d["комиссия_ед"] = comm_pu
    d["прибыль_полн_ед"] = price - full_pu
    # база=план: реализация на проданном = плану поставок
    d["прибыль_ячейка"] = (price - full_pu) * units
    # база=прогноз (опц.): продано = min(спрос_сцен, план); себест/ед та же
    if demand is not None:
        sold = np.array(
            [
                min(demand.get(f"{ch[i]}{int(prod[i])}", units[i]), units[i])
                for i in range(n)
            ]
        )
        d["продано_прог"] = sold
        d["выручка_прог"] = price * sold
        d["прибыль_прог"] = (price - full_pu) * sold

    pools = {
        "материал": float(material.sum()),
        "конверсия": float(conv_pool),
        "канал": float(channel.sum()),
        "накладные": float(oh_pool),
        "смены": float(shifts),
        "машино_часы": float(mh),
        "эксплуатация_станков": float(machine_opex),
        "всего": float(full.sum()),
    }
    return d, pools


def _own_cost_series(
    paths: list[str], group: int, company: int
) -> list[CostReport]:
    """Отчёты затрат СВОЕЙ фирмы (для калибровки yield/цены), компания 1..8."""
    out = []
    for p in paths:
        try:
            c = parse_costs(p)
        except Exception:
            continue
        if c.group == group and c.company == company and 1 <= c.company <= 8:
            out.append(c)
    return out


def _plan_edited(plan: dict[str, float], planned: dict[int, float]) -> bool:
    """Отредактирован ли план решений относительно исполненного (`planned`
    отчёта). Нет плана в решениях → считаем исполненным (False). Порог 2 %
    суммарного отклонения по продуктам (мелкие расхождения — округление/
    капасити-обрезка отчёта)."""
    if not plan:
        return False
    by_prod = {
        p: sum(plan.get(f"{ch}{p}", 0.0) for ch in CHANNELS) for p in PRODUCTS
    }
    tot = sum(planned.values()) or 1.0
    diff = sum(abs(by_prod[p] - planned.get(p, 0.0)) for p in PRODUCTS)
    return diff / tot > 0.02


def cost(
    current: str,
    train: list[str] | None = None,
    history: list[str] | None = None,
    yield_halflife: float = YIELD_HALFLIFE,
    demand: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Себестоимость РЕШЕНИЯ по 9 ячейкам (объём = план поставок, цены/реклама/
    комиссии/смены/станки — с листа 'Your decisions'). Возвращает df (ед/цена/
    выручка, компоненты затрат, себест_полн_ед, CM_ед, прибыль_полн_ед,
    прибыль_ячейка) + df.attrs (meta, пулы, yields, двухрежимная сверка с P&L).
    yield-факторы и спот-цена сырья калибруются по кварталам СВОЕЙ фирмы из
    current+train+history (EWMA). При `demand` (продано по ячейкам, ключи EN
    'EAEU1'..) добавляются реализационные колонки на базе прогноза."""
    cur = parse_costs(current)
    if not 1 <= cur.company <= 8:
        raise ValueError(
            "cost требует отчёт компании 1..8 (свои производственные данные); "
            f"получена компания {cur.company} (history-рамп/half-final)."
        )
    paths = [current] + list(train or []) + list(history or [])
    series = _own_cost_series(paths, cur.group, cur.company)
    yields = calibrate_yields(series, yield_halflife)
    mat_price = material_unit_price(series, cur)
    dec = parse_decisions(current)
    dist_comm: dict[str, float] = dec["dist_comm"]
    # добор текущей комиссии из листа 'W', если в решениях не задана
    _, own = parse_report(current)
    for chn in CHANNELS:
        if chn not in dist_comm:
            v = own.loc[own["channel"] == chn, "dist_comm"].dropna()
            if len(v):
                dist_comm[chn] = float(v.iloc[0])

    # производственные решения: смены, парк станков, реклама
    shifts_override = (
        int(dec["shifts"]) if dec["shifts"] not in (None, 0) else None
    )
    machines = (
        cur.machines + (dec["mach_buy"] or 0.0) - (dec["mach_sell"] or 0.0)
    )
    ad = float(dec["adspend_total"])
    ad_override = ad * 1000.0 if ad > 0 else None

    df = _cell_frame(current, dec["plan"], dec["price"])
    d, pools = _allocate(
        df,
        cur,
        mat_price,
        yields,
        dist_comm,
        dec["assembly_min"],
        ad_override,
        shifts_override,
        machines,
        demand,
    )
    d = d.rename(
        columns={"units": "ед", "price": "цена", "revenue": "выручка"}
    )
    d.insert(0, "продукт", d.pop("product"))
    d.insert(0, "канал", d.pop("channel").map(RU))

    # двухрежимная сверка сметы с фактом P&L: смета (производственный прогон) vs
    # (COGS + накладные). Решения = исполненным → должна сходиться (inventory-
    # шум); отредактированы → форвард-оценка, сверка неприменима.
    edited = _plan_edited(dec["plan"], cur.planned)
    pl_actual = cur.cogs + cur.overhead_total
    d.attrs["meta"] = dict(
        company=cur.company,
        group=cur.group,
        year=cur.year,
        quarter=cur.quarter,
        shifts=int(pools["смены"]),
        mat_price=round(mat_price, 2),
        режим="форвард" if edited else "исполненные",
    )
    d.attrs["yields"] = {k: round(v, 4) for k, v in yields.items()}
    d.attrs["pools"] = {k: round(v, 1) for k, v in pools.items()}
    d.attrs["pl_check"] = dict(
        смета=round(pools["всего"], 0),
        факт_cogs_плюс_накл=round(pl_actual, 0),
        машино_часы_факт=round(cur.machine_worked_h, 0),
        машино_часы_расч=round(pools["машино_часы"], 0),
        применима=not edited,
    )
    return d


def unit_costs(
    current: str,
    train: list[str] | None = None,
    history: list[str] | None = None,
) -> dict[str, dict[str, float]] | None:
    """Лёгкий помощник для forecast: {ячейка(EAEU1..): {full, cm}} на единицу.
    None, если отчёт без производственных данных (компания вне 1..8)."""
    try:
        d = cost(current, train, history)
    except ValueError, KeyError:
        return None
    out: dict[str, dict[str, float]] = {}
    inv = {v: k for k, v in RU.items()}
    for _, r in d.iterrows():
        ch = inv[str(r["канал"])]
        out[f"{ch}{int(r['продукт'])}"] = {
            "full": float(r["себест_полн_ед"]),
            "cm": float(r["CM_ед"]),
        }
    return out
