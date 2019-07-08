# makefile for xspline package

build: setup.py
	python setup.py build

install: setup.py
	python setup.py install

sdist: setup.py
	python setup.py sdist

test:
	python tests/check_utils.py
	python tests/check_bspline.py
	python tests/check_xspline.py

clean:
	rm -rf build dist *.egg-info
	rm -rf xspline/__pycache__
	rm -rf tests/__pycache__