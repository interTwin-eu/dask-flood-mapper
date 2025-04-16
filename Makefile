SHELL = /bin/bash
.PHONY: help clean install test version dist

help:
	@echo "make clean"
	@echo " clean all python build/compilation files and directories"
	@echo "make test"
	@echo " run test with coverage"
	@echo "make version"
	@echo " update _version.py with current version tag"
	@echo "make dist"
	@echo " build the package ready for distribution and update the version tag"

clean:
	find . -name '*.pyc' -exec rm --force {} +
	find . -name '*.pyo' -exec rm --force {} +
	find . -name '*~' -exec rm --force {} +
	rm --force .coverage
	rm --force --recursive .pytest_cache
	rm --force --recursive build/
	rm --force --recursive dist/
	rm --force --recursive *.egg-info

test:
	pytest tests/ -rsx --verbose --color=yes --cov=dask_flood_mapper --cov-report term-missing

version:
	echo -e "__version__ = \"$(shell git describe --always --tags --abbrev=0)\"\n__commit__ = \"$(shell git rev-parse --short HEAD)\"" > src/dask_flood_mapper/_version.py

dist: version
	python3 -m build
