# gmc-forecaster

Двухшаговая модель «доля рынка → спрос» для прогнозирования спроса в
бизнес-симуляции Global Management Challenge (pandas, scikit-learn).

## Установка

Проект использует [uv](https://docs.astral.sh/uv/). Весь код запускается
только внутри виртуального окружения через `uv run` — окружение и сама
утилита ставятся автоматически при первом запуске:

```bash
uv sync
```

## Запуск

Утилита вызывается командой `gmc-forecaster` и имеет три подкоманды.
Справка:

```bash
uv run gmc-forecaster --help
uv run gmc-forecaster <команда> --help
```

### `parse` — разбор отчёта GMC

Читает отчёт(ы) `.xls`/`.xlsx` (лист `W`) и печатает таблицу по каналам ×
продуктам. Можно выгрузить в JSON/CSV, получить текстовый бриф или склеить
несколько отчётов в таблицу переходов `X_t -> demand_next`:

```bash
# таблица в stdout
uv run gmc-forecaster parse W115264.xls

# выгрузка record в JSON / длинной таблицы в CSV
uv run gmc-forecaster parse W115264.xls --json out.json
uv run gmc-forecaster parse W115264.xls --csv  out.csv

# человекочитаемый бриф одного квартала
uv run gmc-forecaster parse W115264.xls --explain

# склеить историю в обучающую таблицу переходов
uv run gmc-forecaster parse data/*.xls --history hist.csv
```

### `forecast` — прогноз спроса под сценарием

Основной рабочий интерфейс «крутить цены/рекламу»: считает спрос на
следующий квартал по всем 9 ячейкам (3 продукта × 3 канала) под заданным
сценарием решений.

```bash
uv run gmc-forecaster forecast \
    --current  W115264.xls \
    --train    W1152*.xls W1312*.xls \
    --history  Hst*.xlsx \
    --scenario scenario.json \
    --out      forecast.csv
```

- `--current` — текущий отчёт (обязательно);
- `--train` — конкурентные отчёты для обучения логит-модели доли (обязательно);
- `--history` — файлы группы 0 для оценки сезонности (опционально);
- `--scenario` — JSON со сценарием (обязательно);
- `--out` — сохранить прогноз в CSV (опционально).

Формат `scenario.json` (незаданные ячейки сохраняют текущие значения):

```json
{
  "quarter_next": 1,
  "price": {"EAEU1": 300, "EAEU3": 820, "INT1": 320},
  "adspend_mult": 1.15
}
```

### `backtest` — бэктест модели спроса

Обучает прозрачную регрессию спроса на истории отчётов и сравнивает её с
наивными базлайнами на отложенном квартале:

```bash
uv run gmc-forecaster backtest data/*.xls --holdout 3
```

`--holdout` — номер квартала для валидации (по умолчанию `3`).

## Разработка

Форматтер, линтер и тайп-чекер запускаются одной командой:

```bash
make validate
```
