#!/bin/sh

cd $(dirname $0)
python3 setup.py build_ext --inplace
