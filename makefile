PYTHON = python3
PIP = pip3

MAIN_SCRIPT = main.py
TRAIN_SCRIPT = train.py
DATASET_SCRIPT = make_dataset.py
REQUIREMENTS = requirements.txt

.PHONY: run train preprocess clean install

run:
	$(PYTHON) $(MAIN_SCRIPT)

preprocess:
	$(PYTHON) $(DATASET_SCRIPT)

train:
	$(PYTHON) $(TRAIN_SCRIPT)

clean:
	rm -rf __pycache__ *.pyc .DS_Store

install:
	$(PIP) install -r $(REQUIREMENTS)
