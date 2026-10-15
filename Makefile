PY ?= .venv/bin/python

.PHONY: test run zed

test:
	$(PY) -m pytest

run:
	$(PY) talk.py

zed:
	$(PY) zed_server.py
