.PHONY: install test data-check features prototypes adapt evaluate plots all clean

CONFIG ?= configs/realworld_to_clipart.yaml

install:
	pip install -e ".[dev]"

test:
	pytest

data-check:
	python scripts/validate_data.py --config $(CONFIG)

features:
	python scripts/extract_features.py --config $(CONFIG)

prototypes:
	python scripts/init_prototypes.py --config $(CONFIG)

adapt:
	python scripts/adapt.py --config $(CONFIG)

evaluate:
	python scripts/evaluate.py --config $(CONFIG)

plots:
	python scripts/make_plots.py --config $(CONFIG)

all:
	python scripts/run_pipeline.py --config $(CONFIG)

clean:
	rm -rf data/cache results/logs results/*.json
