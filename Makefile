.PHONY: format
format:
	python -m black subsetter
	python -m isort subsetter

.PHONY: format-check
format-check:
	python -m black subsetter --check
	python -m isort subsetter --check

.PHONY: mypy
mypy:
	python -m mypy subsetter

.PHONY: pylint
pylint:
	python -m pylint subsetter

.PHONY: lint
lint: format-check mypy pylint
