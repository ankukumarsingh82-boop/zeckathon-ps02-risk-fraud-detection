PY ?= .venv/bin/python
.PHONY: setup all app scrape test clean
setup:
	python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
all:
	$(PY) run_all.py
scrape:
	$(PY) run_all.py --scrape
app:
	.venv/bin/streamlit run app.py
test:
	$(PY) -m pytest -q tests
