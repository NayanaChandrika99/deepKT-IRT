.PHONY: help data train_wdirt train_sakt export demo

PYTHON ?= uv run python
WD_CONFIG ?= configs/wd_irt_edm.yaml
SAKT_CONFIG ?= configs/sakt_assist2009.yaml
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
	@echo "  make data split_seed=<seed>        # materialize deterministic splits and learning events"
	@echo "  make train_wdirt config=<path>     # train Wide & Deep IRT pipeline"
	@echo "  make train_sakt config=<path>      # train SAKT via pyKT"
	@echo "  make export                        # export item and student artifacts"
	@echo "  make demo student_id=...           # run CLI demo joining both engines"

data:
	$(PYTHON) -m src.common.data_pipeline \
		--raw-dir $(raw_dir) \
		--dataset $(dataset) \
		--seed $(split_seed) \
		--events-out $(events_out) \
		--splits-out $(splits_out)

train_wdirt:
	@echo "[wd_irt] Expected command: $(PYTHON) -m src.wd_irt.train --config $(WD_CONFIG)"

train_sakt:
	@echo "[sakt] Expected command: $(PYTHON) -m src.sakt_kt.train --config $(SAKT_CONFIG)"

export:
	@echo "[export] Expected command: $(PYTHON) -m src.common.exporters --item-output reports/item_params.parquet --student-output reports/student_state.parquet"

demo:
	@$(PYTHON) scripts/demo_trace.py --student-id $(student_id) --topic $(topic) --time-window $(time_window)
