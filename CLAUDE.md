# gmc-forecaster

Прогноз спроса (ЗАКАЗЫ) в бизнес-симуляции Global Management Challenge.
Двухшаговая модель «доля рынка → спрос». Единица прогноза — **9 ячеек**:
3 продукта × 3 канала (EAEU/ЕАЭС, ASEAN/АСЕАН, INT/Интернет).

Почему именно share-модель, а не прямой own-demand, и весь дата-анализ,
стоящий за выбором, — в [FINDINGS.md](FINDINGS.md). Там же нюансы данных
(спрос ≠ продажи ≠ доля; доля эндогенна и течёт как признак).
**Важно:** FINDINGS.md — исследовательский журнал, местами отстаёт от кода
(см. «Расхождения»). При конфликте ориентируйся на код.

## Запуск

```bash
uv sync                 # окружение
make validate           # ruff format + ruff check + mypy --strict
uv run gmc-forecaster --help
```

Рабочий сценарий «крутить цены/рекламу»:

```bash
uv run gmc-forecaster \
  --current  data/W115264.xls \
  --train    data/W1152*.xls data/W1312*.xls \
  --history  data/Hst*.xlsx \
  --scenario scenario.json --out forecast.csv
```

`scenario.json`: `{"price": {"EAEU1": 300, ...}, "adspend_mult": 1.15,
"dist_n": {"EAEU": 7, ...}, "dist_comm": {"EAEU": 15, ...},
"dist_elasticity": {"n": 0.15, "comm": 0.10}}` — частичные абс. цены по ячейкам
+ опц. множитель рекламы + опц. **пер-канальные** число/комиссия дистрибьюторов
(EAEU/ASEAN/INT; действуют на все 3 продукта канала). Дистрибьюторы наблюдаемы
только у своей компании → дают множитель собственной привлекательности `A_own`
в логите доли; сила эффекта задаётся коэффициентами `dist_elasticity` (дефолт
`n`=0.15, `comm`=0.10). Незаданное берётся из `--current`. Квартал прогноза =
следующий за кварталом в `--current`.
Колонка **Δ_рычаг_%** изолирует эффект решения (сезонность есть и в базе,
и в сценарии → сокращается).

## Архитектура (`src/gmc_forecaster/`)

- **parser.py** — `parse_report(path)` → `(meta, df по 9 ячейкам)`. Читает
  **только лист `W`** (плоский числовой экспорт) по позиционной схеме `SCHEMA`
  (0-based индексы, `off = 3*(product-1) + ch`). Не зависит от языка и версии
  отчёта. `_num`/`_stars` — нормализация ячеек/рейтинга-звёзд.
- **model.py** — ядро:
  - `load_panel(paths)` → панель по всем 8 компаниям (price/share/adspend/rating),
    + внешняя опция `share_out = 100 − Σ`.
  - `ShareModel` — **стадия 1**: логит доли через инверсию Берри
    `ln(sᵢ/s₀) = β·X + FE(ячейка) + FE(группа)`, OLS (`LinearRegression`).
    Признаки `FEATURES = [log_price, log_adspend, rating]`. `predict_shares`
    даёт MNL-замещение, `counterfactual` — что будет с долями при смене цены.
  - `fit_seasonality(hst)` → сезонные множители {Q:factor} по группе 0.
    `cell_volume` = own_sold/(own_share/100). **Стадия 2**: спрос ≈ доля × объём
    с сезонной поправкой.
- **forecast.py** — `forecast(current, train, history, scenario)`: собирает
  пайплайн, применяет сценарий к своей компании, возвращает df (спрос текущий/
  база/сценарий, доля, Δ_рычаг) + `df.attrs["meta"]`.
- **cli.py** — точка входа `gmc-forecaster` (entry point в pyproject).

## Данные (`data/`)

- `W<gg><c><yy><q>.xls` — отчёт: группа/компания/год/квартал.
  `W1152*` = группа 11 комп.5; `W131*` = группа 13 комп.1.
- `Hst*.xlsx` — history-кварталы группы 0 (компании-клоны) → только сезонность.
- `1Quarter..5Quarter`, `6..10.xls` — отчёты разных версий (RU/PL/ES) для
  проверки версионной устойчивости парсера `W`.
- `gmc_w_schema.json` — выведенная схема листа `W` (справочно).

## Расхождения кода с FINDINGS.md

- Имена `gmc_parser_flat.py` / `gmc_share_model.py` / `gmc_predict.py` устарели —
  это модули пакета выше + CLI `gmc-forecaster`.
- `scenario.json` **не** принимает `quarter_next` (убрано; квартал — из `--current`).
- В коде остался **только** плоский парсер листа `W`; именованного парсера нет.

## Конвенции

- Python ≥3.14, `mypy --strict`, ruff line-length 79. Комментарии/вывод — на русском.
- Весь Python-код запускай только через `uv run`.
