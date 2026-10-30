.PHONY: install install-models lint format test figures eval docker-build
PY ?= python

install:
	$(PY) -m pip install -e ".[dev]"

install-models:
	$(PY) -m pip install -e ".[models,dev]"

lint:
	ruff check . && ruff format --check .

format:
	ruff check . --fix && ruff format .

test:
	$(PY) -m pytest

figures:
	$(PY) scripts/make_figures.py

# Scores existing transcripts in output*/ against eval/data labels; see README "Reproducing the results".
eval:
	$(PY) -m eval.evaluate && $(PY) -m eval.diar_eval && $(PY) -m eval.report

docker-build:
	docker build -t profanity-diarization .
