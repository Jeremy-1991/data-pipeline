SHELL := bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: evaluations
.PHONY: clean

data:
	mkdir data

models:
	mkdir models

data/raw_dataset.csv: | data
	python -m src.load_data

data/clean_dataset.csv: data/raw_dataset.csv
	python -m src.clean_data

data/preprocess_dataset.csv models/preprocessor.pkl: data/clean_dataset.csv | models
	python -m src.preprocess_data

data/train_dataset.csv data/test_dataset.csv models/model.pkl: data/preprocess_dataset.csv
	python -m src.training

evaluations: models/model.pkl data/test_dataset.csv
	python -m src.evaluate

clean:
	rm -rf data models