# Minesweeper RL — Experiment Workflow
# All targets activate .venv automatically if present.
#
# Usage:
#   make train                                          # default config
#   make train CONFIG=experiments/configs/exp_002.yaml
#   make pull                                           # latest run
#   make pull RUN=mw_ppo_5x5x3_seed42_xxx
#   make analyze                                        # latest local run
#   make analyze RUN=mw_ppo_5x5x3_seed42_xxx
#   make analyze RUN=mw_ppo_5x5x3_seed42_xxx EXP_ID=exp_002
#   make eval                                           # batch eval on latest run
#   make eval RUN=mw_ppo_5x5x3_seed42_xxx
#   make play RUN=mw_ppo_5x5x3_seed42_xxx
#   make compare
#   make tensorboard
#   make list
#   make test

# Defaults
CONFIG ?= configs/modal_quick_config.yaml
RUN    ?=
EXP_ID ?=

# Activate venv if it exists
VENV_ACTIVATE = $(if $(wildcard .venv/bin/activate),source .venv/bin/activate &&,)
PYTHON = $(VENV_ACTIVATE) python
MODAL  = $(VENV_ACTIVATE) modal

.PHONY: train pull analyze eval play compare tensorboard list test help

## train: Launch Modal training with CONFIG (default: configs/modal_quick_config.yaml)
train:
	$(MODAL) run train_modal.py --config $(CONFIG)

## pull: Pull RUN (or latest) from Modal Volume to training_runs/
pull:
	$(if $(RUN), \
		bash scripts/pull_run.sh $(RUN), \
		bash scripts/pull_run.sh \
	)

## analyze: Analyze RUN (or latest) TensorBoard logs; optionally save JSON with EXP_ID
analyze:
	$(PYTHON) scripts/analyze.py \
		$(if $(RUN),$(RUN),) \
		$(if $(EXP_ID),--exp-id $(EXP_ID),)

## eval: Batch evaluate RUN (100 episodes) — clean win rate for experiment comparison
eval:
	$(PYTHON) play.py --mode batch --num_episodes 100 \
		$(if $(RUN),--model_dir training_runs/$(RUN),--training_run_dir training_runs/)

## play: Watch agent play with visualization
play:
	$(PYTHON) play.py --mode agent \
		$(if $(RUN),--model_dir training_runs/$(RUN),--training_run_dir training_runs/)

## compare: Compare all runs in training_runs/ (50 episodes each)
compare:
	$(PYTHON) play.py --mode compare --num_episodes 50 --training_run_dir training_runs/

## test: Run all tests
test:
	$(PYTHON) -m pytest tests/ -v

## tensorboard: Launch TensorBoard on training_runs/
tensorboard:
	$(VENV_ACTIVATE) tensorboard --logdir training_runs/

## list: List all runs in Modal Volume
list:
	$(MODAL) volume ls minesweeper-runs /

## help: Show this help message
help:
	@grep -E '^## ' Makefile | sed 's/## /  make /'
