# ABOUTME: Explains how raw, interim, and processed data are organized.
# ABOUTME: Captures dataset download steps plus deterministic split expectations.

## Data Layout

- `raw/` holds immutable vendor datasets such as EDM Cup 2023 clickstream and ASSISTments2009. Store README pointers, checksums, or download scripts rather than the files themselves.
- `interim/` stores canonical learning events after cleaning, enrichment, and schema normalization. Filenames follow `<dataset>_<seed>_<timestamp>_events.parquet`.
- `processed/` contains model-ready tensors or tables per config. Use `<engine>_<dataset>_<config_name>.parquet`.

## Downloading Datasets

1. **EDM Cup 2023**: request access via the EDM Cup portal, download the clickstream zip, and extract it into `raw/edm_cup_2023/`. Record checksums in `raw/edm_cup_2023/sha256sum.txt` (see sample listing below).
2. **ASSISTments Skill Builder (2009)**: download `skill_builder_data.csv` from the ASSISTments data release, place it in `raw/assistments_skill_builder/`, and capture its checksum with `shasum -a 256 skill_builder_data.csv > sha256sum.txt`.

Automate downloads via scripts placed under `scripts/` and reference them within `Makefile data`. For EDM Cup, the raw directory should contain:

| File | Description |
| --- | --- |
| `action_logs.csv` | Clickstream events per assignment |
| `assignment_details.csv` | Maps assignment logs to teacher/class/student identifiers |
| `assignment_relationships.csv` | Links sequences/assignments |
| `problem_details.csv` | Problem metadata and skill tags |
| `sequence_details.csv` | Curriculum structure for sequences |
| `sequence_relationships.csv` | Sequence linkage metadata |
| `hint_details.csv`, `explanation_details.csv` | Support content metadata |
| `training_unit_test_scores.csv`, `evaluation_unit_test_scores.csv` | Assessment labels |
| `sha256sum.txt` | Checksums generated with `shasum -a 256 *.csv` |

## Split Protocol

- All splits are deterministic and student-based. Use a configurable seed (default `42`) to assign students to `train`, `val`, and `test`.
- Persist split manifests under `data/splits/<dataset>_<seed>.json` containing `{"train": [...], "val": [...], "test": [...]}`.
- Reuse the same manifest for both WD-IRT and SAKT to keep evaluation aligned.

## Learning Event Schema

Normalize every record to include:

| Column | Description |
| --- | --- |
| `user_id` | Stable student identifier |
| `item_id` | Item or question identifier |
| `skill_ids` | List of KC tags |
| `timestamp` | ISO8601 string |
| `correct` | 0/1 |
| `action_sequence_id` | Clickstream session id |
| `latency_ms` | Time from first action to submission |
| `help_requested` | Boolean flag |

Document any additional fields inside this README as you extend the schema.
For ASSISTments, the raw directory should contain:

| File | Description |
| --- | --- |
| `skill_builder_data.csv` | Interaction-level events (columns include `order_id`, `user_id`, `problem_id`, `skill_id`, `correct`, `ms_first_response`, `hint_total`, etc.) |
| `sha256sum.txt` | Checksum generated via `shasum -a 256 skill_builder_data.csv` |
