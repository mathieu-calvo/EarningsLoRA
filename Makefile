.PHONY: install install-dev test test-all lint format prepare train evaluate publish app clean

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,train,eval,app]"

test:
	pytest -m "not slow" -v

test-all:
	pytest -v

lint:
	ruff check src tests

format:
	ruff format src tests
	ruff check --fix src tests

prepare:
	python scripts/prepare_dataset.py

train:
	python scripts/train.py

evaluate:
	python scripts/evaluate.py

publish:
	python scripts/publish.py

app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
