.PHONY: bootstrap test lint format-check compile schema checksums doctor doctor-live prepare smoke headline quality package verify-archive clean

bootstrap:
	./scripts/bootstrap.sh

test:
	.venv/bin/pytest

lint:
	.venv/bin/ruff check src tests scripts

format-check:
	.venv/bin/ruff format --check src tests scripts

compile:
	.venv/bin/python -m compileall -q src tests scripts

schema:
	PYTHONPATH=src .venv/bin/python scripts/generate_schema.py

checksums:
	.venv/bin/python scripts/update_checksums.py

doctor:
	.venv/bin/bc250 doctor --config configs/smoke.yaml

doctor-live:
	.venv/bin/bc250 doctor --config configs/smoke.yaml --live

prepare:
	.venv/bin/bc250 prepare --config configs/headline.yaml

smoke:
	.venv/bin/bc250 run --config configs/smoke.yaml --limit 1

headline:
	.venv/bin/bc250 headline --config configs/headline.yaml --yes

quality:
	.venv/bin/bc250 run --config configs/quality.yaml

package:
	./scripts/package.sh

verify-archive:
	./scripts/verify_archive.sh ../browsecomp-250-openai-compatible.zip

clean:
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info src/browsecomp250.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
