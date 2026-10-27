.DEFAULT_GOAL := sota
PYTHON ?= python3
.PHONY: sota check-sota test-sota
sota:
	$(PYTHON) -m recipes.sota6b show
check-sota:
	$(PYTHON) -m recipes.sota6b check
	$(PYTHON) experiments/kd6b-r5-20261020/launch.py check
	bash -n slurm/kd6b-r5-20261020.sbatch
test-sota:
	$(PYTHON) -m unittest discover -s tests -p test_sota_recipe.py -v
	$(PYTHON) experiments/kd6b-r5-20261020/test_contracts.py
