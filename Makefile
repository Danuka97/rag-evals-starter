PYTHON ?= python

.PHONY: setup test run lint format clean

# Create a virtual environment and install dependencies
setup:
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

# Run all unit tests
test:
	$(PYTHON) -m pytest -q

# Execute the pipeline on the sample data
run:
	$(PYTHON) src/pipeline.py --input data/sample_input.json --output output/metrics.csv

# Lint the code with ruff
lint:
	ruff src tests

# Format code with black
format:
	black src tests

# Remove build artefacts
clean:
	rm -rf .venv __pycache__ output