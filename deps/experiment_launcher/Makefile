package: 
	python setup.py sdist

install: package
	pip install $(shell ls dist/*.tar.gz)

all: clean install

upload:
	python -m twine upload dist/*

clean:
	rm -rf dist
	rm -rf build
