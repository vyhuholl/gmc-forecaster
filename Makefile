.PHONY: format lint type-check validate

RUN_CMD := uv run

format: 
	$(RUN_CMD) ruff format .

lint:
	$(RUN_CMD) ruff check .

type-check:
	$(RUN_CMD) mypy .

validate: format lint type-check