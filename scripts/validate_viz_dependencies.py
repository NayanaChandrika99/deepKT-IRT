"""
ABOUTME: Validates data dependencies for system_demo.ipynb notebook
ABOUTME: Checks file existence, schemas, row counts, and data quality
"""

import pandas as pd
from pathlib import Path
import sys

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_file(file_path, expected_rows=None, required_cols=None, optional=False):
    """Validate a single parquet file."""
    status = []

    # Check existence
    if not file_path.exists():
        if optional:
            status.append(f"{YELLOW}⚠ OPTIONAL{RESET} {file_path} - NOT FOUND")
            return status, None
        else:
            status.append(f"{RED}✗ REQUIRED{RESET} {file_path} - NOT FOUND")
            return status, None

    # Load and validate
    try:
        df = pd.read_parquet(file_path)
        size_mb = file_path.stat().st_size / (1024 * 1024)

        status.append(f"{GREEN}✓{RESET} {file_path.name}")
        status.append(f"  Size: {size_mb:.1f} MB")
        status.append(f"  Rows: {len(df):,}")
        status.append(f"  Columns: {len(df.columns)}")

        # Check row count expectations
        if expected_rows:
            if isinstance(expected_rows, tuple):
                min_rows, max_rows = expected_rows
                if not (min_rows <= len(df) <= max_rows):
                    status.append(f"  {YELLOW}⚠{RESET} Expected {min_rows:,}-{max_rows:,} rows, got {len(df):,}")
            else:
                if abs(len(df) - expected_rows) > expected_rows * 0.1:  # 10% tolerance
                    status.append(f"  {YELLOW}⚠{RESET} Expected ~{expected_rows:,} rows, got {len(df):,}")

        # Check required columns
        if required_cols:
            missing = set(required_cols) - set(df.columns)
            if missing:
                status.append(f"  {RED}✗ Missing columns{RESET}: {missing}")
            else:
                status.append(f"  {GREEN}✓{RESET} All required columns present")

        return status, df

    except Exception as e:
        status.append(f"{RED}✗ ERROR{RESET} loading {file_path}: {e}")
        return status, None

def main():
    print("=" * 80)
    print("DATA DEPENDENCY VALIDATION FOR system_demo.ipynb")
    print("=" * 80)
    print()

    base_dir = Path(__file__).parent.parent
    reports_dir = base_dir / "reports"
    data_dir = base_dir / "data" / "interim"

    all_valid = True
    dataframes = {}

    # 1. Core: bandit_decisions.parquet
    print("1. CORE FILE: bandit_decisions.parquet")
    print("-" * 80)
    required_cols = [
        't', 'student_id', 'skill_id', 'candidate_item_id',
        'mu', 'sigma', 'ucb', 'alpha', 'mode',
        'rank', 'chosen'
    ]
    status, df = check_file(
        reports_dir / "bandit_decisions.parquet",
        expected_rows=(13, 50000),
        required_cols=required_cols
    )
    for line in status:
        print(line)
    if df is not None:
        dataframes['decisions'] = df
        # Additional validation
        if 'chosen' in df.columns:
            chosen_per_decision = df.groupby(['t', 'student_id', 'skill_id'])['chosen'].sum()
            if not (chosen_per_decision == 1).all():
                print(f"  {RED}✗ INVARIANT VIOLATION{RESET}: Not exactly one chosen item per decision")
                all_valid = False
            else:
                print(f"  {GREEN}✓{RESET} Invariant: Exactly one chosen item per decision")
    else:
        all_valid = False
    print()

    # 2. skill_mastery.parquet
    print("2. CONTEXT: skill_mastery.parquet")
    print("-" * 80)
    status, df = check_file(
        reports_dir / "skill_mastery.parquet",
        expected_rows=50000,
        required_cols=['mastery_mean']
    )
    for line in status:
        print(line)
    if df is not None:
        dataframes['skill_mastery'] = df
    else:
        all_valid = False
    print()

    # 3. item_params.parquet
    print("3. CONTEXT: item_params.parquet")
    print("-" * 80)
    status, df = check_file(
        reports_dir / "item_params.parquet",
        expected_rows=1835,
        required_cols=['difficulty', 'discrimination']
    )
    for line in status:
        print(line)
    if df is not None:
        dataframes['item_params'] = df
    else:
        all_valid = False
    print()

    # 4. edm_cup events
    print("4. CONTEXT: edm_cup_2023_42_events.parquet")
    print("-" * 80)
    status, df = check_file(
        data_dir / "edm_cup_2023_42_events.parquet",
        expected_rows=5167603,
        required_cols=['user_id', 'latency_ms']
    )
    for line in status:
        print(line)
    if df is not None:
        dataframes['events'] = df
    else:
        all_valid = False
    print()

    # 5. sakt_student_state.parquet
    print("5. CONTEXT: sakt_student_state.parquet")
    print("-" * 80)
    status, df = check_file(
        reports_dir / "sakt_student_state.parquet",
        expected_rows=1000000
    )
    for line in status:
        print(line)
    if df is not None:
        dataframes['student_state'] = df
    else:
        all_valid = False
    print()

    # 6. sakt_attention.parquet (OPTIONAL)
    print("6. OPTIONAL: sakt_attention.parquet")
    print("-" * 80)
    status, df = check_file(
        reports_dir / "sakt_attention.parquet",
        expected_rows=50000,
        optional=True
    )
    for line in status:
        print(line)
    if df is not None:
        dataframes['attention'] = df
    print()

    # Cross-file validation
    if 'decisions' in dataframes and 'item_params' in dataframes:
        print("7. CROSS-FILE VALIDATION")
        print("-" * 80)

        # Check if candidate_item_ids reference valid items
        decision_items = set(dataframes['decisions']['candidate_item_id'].unique())
        param_items = set(dataframes['item_params'].index if hasattr(dataframes['item_params'].index, 'name') else dataframes['item_params'].iloc[:, 0])

        # Try to find item ID column
        item_cols = [col for col in dataframes['item_params'].columns if 'item' in col.lower() or 'id' in col.lower()]
        if item_cols:
            param_items = set(dataframes['item_params'][item_cols[0]].unique())

        print(f"  Decision items: {len(decision_items)} unique")
        print(f"  Item params: {len(param_items)} unique")

        if decision_items and param_items:
            missing = decision_items - param_items
            if missing and len(missing) < 10:
                print(f"  {YELLOW}⚠{RESET} {len(missing)} decision items not in item_params: {list(missing)[:5]}")
            elif missing:
                print(f"  {YELLOW}⚠{RESET} {len(missing)} decision items not in item_params")
        print()

    # Summary
    print("=" * 80)
    if all_valid:
        print(f"{GREEN}✓ ALL REQUIRED DEPENDENCIES VALID{RESET}")
        print("Ready to build system_demo.ipynb")
        return 0
    else:
        print(f"{RED}✗ VALIDATION FAILED{RESET}")
        print("Fix missing/invalid files before proceeding")
        return 1

if __name__ == "__main__":
    sys.exit(main())
