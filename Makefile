PY ?= python

.PHONY: all install test ghz vqe quick clean

all: install test ghz vqe

install:
	$(PY) -m pip install -r requirements.txt

test:
	$(PY) -m pytest -q

ghz:
	$(PY) src/ghz_benchmark.py

vqe:
	$(PY) src/h2_vqe.py

# Reduced versions of both experiments (< 2 min on CPU) -> results/quick/
quick:
	$(PY) src/ghz_benchmark.py --quick
	$(PY) src/h2_vqe.py --quick

clean:
	rm -rf results/quick .pytest_cache
