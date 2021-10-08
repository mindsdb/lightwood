#!/usr/bin/env bash

# 2021.09.14

# Build HTML files; run this on the source directory
# Of the form: sphinx-build -b <buildtype> <sourcedir> <builddir>
sphinx-build -b html source build

# TODO: Hack to move static folders - this should be fixed (NS)
cp -r source/tutorials build