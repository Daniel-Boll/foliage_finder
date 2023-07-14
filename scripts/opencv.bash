#!/bin/bash

_pip=poetry
# _pip=pipenv
curl -sSL https://raw.githubusercontent.com/bschnurr/python-type-stubs/add-opencv/cv2/__init__.pyi \
    -o $($_pip run python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')/cv2.pyi
curl -sSL https://raw.githubusercontent.com/bschnurr/python-type-stubs/add-opencv/cv2/__init__.pyi \
    -o $($_pip run python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')/__init__.pyi
unset _pip
