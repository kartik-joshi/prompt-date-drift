PY ?= .venv/bin/python
MODELS ?= qwen2.5:3b-instruct llama3.2:3b

.PHONY: setup pull questions run score plot test all

setup:
	uv venv -p 3.11 .venv
	uv pip install --python $(PY) -e ".[dev]"

pull:
	for m in $(MODELS); do ollama pull $$m; done

questions:
	mkdir -p data
	$(PY) gen_questions.py --seed 0 > data/questions.jsonl

run:
	$(PY) run.py --models $(MODELS)

score:
	$(PY) score.py results/results.jsonl --by-kind --json results/summary.json

plot:
	$(PY) plot.py results/results.jsonl --out results/flip_rate.png

test:
	$(PY) -m pytest -q

all: run score plot
