.PHONY: help data train_wdirt train_sakt export_wdirt export_sakt export demo

PYTHON ?= uv run python
WD_CONFIG ?= configs/wd_irt_edm.yaml
SAKT_CONFIG ?= configs/sakt_assist2009.yaml
WD_CHECKPOINT ?= reports/checkpoints/wd_irt_edm/latest.ckpt
SAKT_CHECKPOINT ?= reports/checkpoints/sakt_edm/sakt_edm_seed42_best.pt
split_seed ?= 42
dataset ?= edm_cup_2023
raw_dir ?= data/raw/$(dataset)
events_out ?= data/interim/$(dataset)_$(split_seed)_events.parquet
splits_out ?= data/splits/$(dataset)_$(split_seed).json
student_id ?= demo_student
topic ?= demo_topic
time_window ?= recent

help:
	@echo "Targets:"
	@echo "  make data split_seed=<seed>                # build canonical events + splits"
	@echo "  make train_wdirt WD_CONFIG=<cfg>           # train Wide & Deep IRT"
	@echo "  make train_sakt SAKT_CONFIG=<cfg>          # train SAKT via pyKT"
	@echo "  make export_wdirt WD_CHECKPOINT=<ckpt>     # export item health artifacts"
	@echo "  make export_sakt SAKT_CHECKPOINT=<pt>      # export student mastery artifacts"
	@echo "  make export                                # run both export targets"
	@echo "  make demo student_id=... topic=...         # run CLI demo joining both engines"

data:
	$(PYTHON) -m src.common.data_pipeline \
		--raw-dir $(raw_dir) \
		--dataset $(dataset) \
		--seed $(split_seed) \
		--events-out $(events_out) \
		--splits-out $(splits_out)

train_wdirt:
	$(PYTHON) -m src.wd_irt.train --config $(WD_CONFIG)

train_sakt:
	$(PYTHON) -m src.sakt_kt.train --config $(SAKT_CONFIG)

export_wdirt:
	$(PYTHON) -m src.wd_irt.train export --config $(WD_CONFIG) --checkpoint $(WD_CHECKPOINT) --output-dir reports

export_sakt:
	$(PYTHON) -m src.sakt_kt.train export --config $(SAKT_CONFIG) --checkpoint $(SAKT_CHECKPOINT) --output-dir reports

export: export_wdirt export_sakt

demo:
	@$(PYTHON) scripts/demo_trace.py --student-id $(student_id) --topic $(topic) --time-window $(time_window)
