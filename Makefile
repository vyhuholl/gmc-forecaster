.PHONY: format lint type-check test validate

RUN_CMD := uv run

format:
	$(RUN_CMD) ruff format .

lint:
	$(RUN_CMD) ruff check .

type-check:
	$(RUN_CMD) mypy .

test:
	$(RUN_CMD) pytest -q

validate: format lint type-check test