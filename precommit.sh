#!/bin/sh -eu
# Run this script before committing to get cleaner jupyter diffs!
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace P2.ipynb
