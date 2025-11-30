#!/usr/bin/env python3
# ABOUTME: Utility script to export small JSON samples for the GitHub Pages site.

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "docs" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPORTS = {
    "attention_sample.json": ROOT / "reports" / "sakt_attention.parquet",
    "skill_sample.json": ROOT / "reports" / "skill_mastery.parquet",
    "pipeline_sample.json": ROOT / "data" / "interim" / "edm_cup_2023_42_events.parquet",
}


def main() -> None:
    for name, source in EXPORTS.items():
        if not source.exists():
            print(f"Skipping {name}: missing {source}")
            continue
        df = pd.read_parquet(source).head(200)
        (OUTPUT_DIR / name).write_text(df.to_json(orient="records", indent=2))
        print(f"Wrote {name} from {source}")


if __name__ == "__main__":
    main()
