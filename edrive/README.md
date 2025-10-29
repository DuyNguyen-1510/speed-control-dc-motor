# edrive


Monorepo skeleton for electric drive simulations in Python (Cursor-friendly).


## Goals
- Compose **battery → boost → DC bus → H-bridge → PMDC motor**
- Add **controllers** (bus/current/speed) and **observers**
- Keep interfaces stable, tests fast, and configs declarative


## Getting started
```bash
pip install -r requirements.txt
pip install -e .
make test
