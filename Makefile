PYTHON ?= python

# Default series length for benchmark
N ?= 2000000

.PHONY: help benchmark

help:
	@echo "Available targets:"
	@echo "  benchmark        Run integer detection benchmark (size N=$(N))"
	@echo "Usage: make benchmark [N=5000000]"

benchmark:
	$(PYTHON) scripts/benchmark_integer_checks.py $(N)
