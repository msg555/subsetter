PYTHON ?= python3

.PHONY: format
format:
	$(PYTHON) -m black subsetter tests
	$(PYTHON) -m isort subsetter tests

.PHONY: format-check
format-check:
	$(PYTHON) -m black subsetter tests --check
	$(PYTHON) -m isort subsetter tests --check

.PHONY: mypy
mypy:
	$(PYTHON) -m mypy subsetter tests

.PHONY: pylint
pylint:
	$(PYTHON) -m pylint subsetter tests

.PHONY: lint
lint: format-check mypy pylint

.PHONY: test
test:
	$(PYTHON) -m pytest -sv tests/

.PHONY: build
build:
	$(PYTHON) -m build

.PHONY: clean
clean:
	rm -rf build dist *.egg-info

.PHONY: pypi-test
pypi-test: build
	TWINE_USERNAME=__token__ TWINE_PASSWORD="$(shell gpg -d test.pypi-token.gpg)" \
    $(PYTHON) -m twine upload --repository testpypi dist/*

.PHONY: pypi-live
pypi-live: build
	TWINE_USERNAME=__token__ TWINE_PASSWORD="$(shell gpg -d live.pypi-token.gpg)" \
    $(PYTHON) -m twine upload dist/*

