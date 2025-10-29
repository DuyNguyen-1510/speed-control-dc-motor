.PHONY: setup lint format test run_boost run_motor run_system


setup:
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install -e .


lint:
ruff check src tests


format:
ruff format src tests
echo "Formatted."


test:
pytest -q


run_boost:
python experiments/run_boost.py -c configs/boost_only.yaml


run_motor:
python experiments/run_motor.py -c configs/motor_only.yaml


run_system:
python experiments/run_system_speed.py -c configs/system_speed_loop.yaml