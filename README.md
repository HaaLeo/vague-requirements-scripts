# Vague-requirements-scripts
This repository contains all the helper and scripts used to evaluate data for the [Vague Requirements Master's Thesis](https://github.com/HaaLeo/Vague-Requirements).

## Installation

```zsh
git clone git@github.com:HaaLeo/vague-requirements-scripts.git
cd vague-requirements-scripts

python3 -m venv ./.venv
source .venv/bin/activate
pip3 install -r requirements-dev.txt
```

## Usage
In the scripts adjust the file paths to your data sets then you can run them like
```zsh
python3 scripts/compare.py
```
