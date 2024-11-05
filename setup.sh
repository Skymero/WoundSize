#!/bin/bash

# Install dependencies from requirements.txt
pip install -r requirements.txt

pip install -r Deepskin/requirements.txt

pip install -r Deepskin/docs/requirements.txt

# python3.10 -m pip install .
cd Deepskin
# Install Deepskin package
# python3.10 ./Deepskin/setup.py
python -m pip install --editable .

cd ..