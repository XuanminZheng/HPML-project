#!/bin/bash

# baseline
python api.py --config baseline --port 8000

# INT4
python api.py --config int4 --port 8001

python api.py --config activation --port 8003
