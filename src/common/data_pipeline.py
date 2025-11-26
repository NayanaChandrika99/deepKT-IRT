# ABOUTME: Builds canonical learning events from raw clickstream datasets.
# ABOUTME: Provides CLI helpers to normalize data and emit deterministic splits.

import json
import random
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pv
import typer

RESPONSE_ACTIONS = {"correct_response", "wrong_response"}
HELP_ACTIONS = {
    "hint_requested",
    "answer_requested",
    "explanation_requested",
    "skill_related_video_requested",
    "live_tutor_requested",
}
PROBLEM_START_ACTION = "problem_started"
RELEVANT_ACTIONS = RESPONSE_ACTIONS | HELP_ACTIONS | {PROBLEM_START_ACTION}
EDM_DATASET = "edm_cup_2023"
ASSISTMENTS_DATASET = "assistments_skill_builder"

app = typer.Typer(help="Normalize datasets into canonical learning events.")


def prepare_learning_events(raw_dir: Path, dataset: str) -> pd.DataFrame:
    """
    Build canonical learning events for the requested dataset identifier.
    """

    normalized = dataset.strip().lower()
    if normalized == EDM_DATASET:
        return _prepare_edm_events(raw_dir)
    if normalized == ASSISTMENTS_DATASET:
        return _prepare_assistments_events(raw_dir)
    raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {EDM_DATASET}, {ASSISTMENTS_DATASET}.")


def generate_user_splits(
    user_ids: Sequence[str], train_ratio: float, val_ratio: float, seed: int
) -> Dict[str, List[str]]:
    """
    Deterministically split unique user identifiers into train/val/test buckets.
    """

    unique_users = sorted({uid for uid in user_ids if uid})
    total = len(unique_users)
    if total == 0:
        return {"train": [], "val": [], "test": []}

    rng = random.Random(seed)
    rng.shuffle(unique_users)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    remaining = total - train_count - val_count

    # Ensure test split keeps at least one user whenever possible.
    if remaining <= 0 and total > 1:
        deficit = 1 - remaining
        while deficit > 0 and val_count > 0:
            val_count -= 1
            deficit -= 1
        while deficit > 0 and train_count > 0:
            train_count -= 1
            deficit -= 1
        remaining = total - train_count - val_count
    test_count = remaining

    # Guard against pathological ratios on very small cohorts.
    if train_count < 0:
        train_count = 0
    if val_count < 0:
        val_count = 0
    if test_count < 0:
        test_count = 0

    train = unique_users[:train_count]
    val = unique_users[train_count : train_count + val_count]
    test = unique_users[train_count + val_count : train_count + val_count + test_count]

    # When user count is tiny, ensure all users are assigned.
    remainder = set(unique_users) - set(train) - set(val) - set(test)
    for uid in remainder:
        test.append(uid)

    return {"train": train, "val": val, "test": test}


@app.command()
def build(
    raw_dir: Path = typer.Option(..., exists=True, file_okay=False, help="Path to dataset root."),
    dataset: str = typer.Option("edm_cup_2023", help="Dataset identifier."),
    seed: int = typer.Option(42, help="Random seed for splits."),
    events_out: Path = typer.Option(..., help="Output parquet for canonical events."),
    splits_out: Path = typer.Option(..., help="Output JSON path for user splits."),
    train_ratio: float = typer.Option(0.7, help="Proportion of users in train split."),
    val_ratio: float = typer.Option(0.15, help="Proportion of users in validation split."),
) -> None:
    typer.echo(f"[data] Building canonical events from {raw_dir} for dataset='{dataset}'")
    try:
        events = prepare_learning_events(raw_dir, dataset)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--dataset") from exc

    events_out.parent.mkdir(parents=True, exist_ok=True)
    splits_out.parent.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[data] Writing {len(events)} events to {events_out}")
    events.to_parquet(events_out, index=False)

    splitter = generate_user_splits(events["user_id"].tolist(), train_ratio, val_ratio, seed)
    typer.echo(f"[data] Train={len(splitter['train'])} Val={len(splitter['val'])} Test={len(splitter['test'])}")
    splits_out.write_text(json.dumps(splitter, indent=2), encoding="utf-8")


def _load_action_logs(path: Path) -> pd.DataFrame:
    convert_options = pv.ConvertOptions(
        include_columns=["assignment_log_id", "timestamp", "problem_id", "action"],
        column_types={
            "assignment_log_id": pa.string(),
            "timestamp": pa.float64(),
            "problem_id": pa.string(),
            "action": pa.string(),
        },
    )
    table = pv.read_csv(path, convert_options=convert_options, read_options=pv.ReadOptions(block_size=1 << 22))
    mask = pc.is_in(table["action"], value_set=pa.array(sorted(RELEVANT_ACTIONS)))
    table = table.filter(mask)
    df = table.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp", "assignment_log_id"])
    df = df.sort_values(["assignment_log_id", "problem_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    df["has_help"] = df["action"].isin(HELP_ACTIONS)
    by_problem = df.groupby(["assignment_log_id", df["problem_id"].fillna("__NO_PROBLEM__")], sort=False, group_keys=False)
    df["help_so_far"] = by_problem["has_help"].cumsum().astype("int64")
    return df


def _load_assignment_details(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "assignment_log_id": "string",
            "student_id": "string",
            "sequence_id": "string",
        },
        usecols=["assignment_log_id", "student_id", "sequence_id"],
    )


def _load_problem_details(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        dtype={
            "problem_id": "string",
            "problem_skill_code": "string",
        },
        usecols=["problem_id", "problem_skill_code"],
    )


def _extract_response_events(action_logs: pd.DataFrame) -> pd.DataFrame:
    responses = action_logs[action_logs["action"].isin(RESPONSE_ACTIONS)].copy()
    responses = responses.dropna(subset=["problem_id"])

    responses["correct"] = (responses["action"] == "correct_response").astype("int64")
    responses["help_requested"] = responses["help_so_far"] > 0

    start_times = (
        action_logs[action_logs["action"] == PROBLEM_START_ACTION]
        .dropna(subset=["problem_id"])
        .drop_duplicates(subset=["assignment_log_id", "problem_id"], keep="first")
        .rename(columns={"timestamp": "problem_start_ts"})
        [["assignment_log_id", "problem_id", "problem_start_ts"]]
    )

    responses = responses.merge(
        start_times,
        on=["assignment_log_id", "problem_id"],
        how="left",
    )
    responses["latency_ms"] = (
        (responses["timestamp"] - responses["problem_start_ts"]).dt.total_seconds() * 1000.0
    ).round().astype("Int64")
    responses["latency_ms"] = responses["latency_ms"].where(responses["latency_ms"] >= 0)
    responses = responses.drop(columns=["has_help", "help_so_far", "problem_start_ts"], errors="ignore")
    return responses


def _attach_assignment_metadata(responses: pd.DataFrame, assignment_details: pd.DataFrame) -> pd.DataFrame:
    merged = responses.merge(
        assignment_details,
        on="assignment_log_id",
        how="left",
        validate="many_to_one",
    )
    merged = merged.dropna(subset=["student_id"])
    return merged


def _attach_problem_metadata(responses: pd.DataFrame, problem_details: pd.DataFrame) -> pd.DataFrame:
    merged = responses.merge(problem_details, on="problem_id", how="left", validate="many_to_one")
    merged["skill_ids"] = merged["problem_skill_code"].map(_to_skill_list)
    merged["skill_ids"] = merged["skill_ids"].apply(lambda skills: skills or [])
    merged = merged.drop(columns=["problem_skill_code"])
    return merged


def _to_skill_list(raw_value: str) -> List[str]:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return []
    text_value = str(raw_value).strip()
    if not text_value:
        return []
    parts = [part.strip() for part in text_value.replace(";", ",").split(",") if part.strip()]
    return parts or [text_value]


def _prepare_edm_events(raw_dir: Path) -> pd.DataFrame:
    raw_dir = raw_dir.resolve()
    action_logs = _load_action_logs(raw_dir / "action_logs.csv")
    assignment_details = _load_assignment_details(raw_dir / "assignment_details.csv")
    problem_details = _load_problem_details(raw_dir / "problem_details.csv")

    responses = _extract_response_events(action_logs)
    responses = _attach_assignment_metadata(responses, assignment_details)
    responses = _attach_problem_metadata(responses, problem_details)

    events = pd.DataFrame(
        {
            "user_id": responses["student_id"],
            "item_id": responses["problem_id"],
            "skill_ids": responses["skill_ids"],
            "timestamp": responses["timestamp"],
            "correct": responses["correct"],
            "action_sequence_id": responses["assignment_log_id"],
            "latency_ms": responses["latency_ms"],
            "help_requested": responses["help_requested"],
        }
    )

    events = events.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    return events


def _prepare_assistments_events(raw_dir: Path) -> pd.DataFrame:
    csv_path = (raw_dir / "skill_builder_data.csv").resolve()
    convert_options = pv.ConvertOptions(
        include_columns=[
            "order_id",
            "assignment_id",
            "user_id",
            "assistment_id",
            "problem_id",
            "correct",
            "ms_first_response",
            "hint_count",
            "hint_total",
            "bottom_hint",
            "skill_id",
            "skill_name",
        ],
        column_types={
            "order_id": pa.float64(),
            "assignment_id": pa.string(),
            "user_id": pa.string(),
            "assistment_id": pa.string(),
            "problem_id": pa.string(),
            "correct": pa.float64(),
            "ms_first_response": pa.float64(),
            "hint_count": pa.float64(),
            "hint_total": pa.float64(),
            "bottom_hint": pa.float64(),
            "skill_id": pa.string(),
            "skill_name": pa.string(),
        },
    )
    table = pv.read_csv(csv_path, convert_options=convert_options, read_options=pv.ReadOptions(block_size=1 << 22))
    df = table.to_pandas()
    df = df.dropna(subset=["user_id", "problem_id"])

    df["timestamp"] = _assistments_timestamp(df["order_id"])
    df["user_id"] = df["user_id"].astype("string")
    df["item_id"] = df["problem_id"].astype("string")
    df["action_sequence_id"] = df["assignment_id"].fillna(df["user_id"]).astype("string")
    df["latency_ms"] = pd.to_numeric(df["ms_first_response"], errors="coerce").round().astype("Int64")
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype("int64").clip(0, 1)

    hint_count = pd.to_numeric(df["hint_count"], errors="coerce").fillna(0)
    hint_total = pd.to_numeric(df["hint_total"], errors="coerce").fillna(0)
    bottom_hint = pd.to_numeric(df["bottom_hint"], errors="coerce").fillna(0)
    df["help_requested"] = (hint_count > 0) | (hint_total > 0) | (bottom_hint > 0)

    df["skill_ids"] = [
        _assistments_skill_list(skill_id_value, skill_name_value)
        for skill_id_value, skill_name_value in zip(df["skill_id"], df["skill_name"])
    ]

    events = pd.DataFrame(
        {
            "user_id": df["user_id"],
            "item_id": df["item_id"],
            "skill_ids": df["skill_ids"],
            "timestamp": df["timestamp"],
            "correct": df["correct"],
            "action_sequence_id": df["action_sequence_id"],
            "latency_ms": df["latency_ms"],
            "help_requested": df["help_requested"].astype(bool),
        }
    )

    events = events.sort_values(["user_id", "timestamp"], kind="mergesort").reset_index(drop=True)
    return events


def _assistments_timestamp(order_ids: pd.Series) -> pd.Series:
    base = pd.Timestamp("2009-01-01", tz="UTC")
    numeric_order = pd.to_numeric(order_ids, errors="coerce").fillna(0).astype("int64")
    return base + pd.to_timedelta(numeric_order, unit="s")


def _assistments_skill_list(skill_id_value: str, skill_name_value: str) -> List[str]:
    ids = _to_skill_list(skill_id_value)
    if ids:
        return ids
    if isinstance(skill_name_value, str) and skill_name_value.strip():
        return [skill_name_value.strip()]
    return []


def main():
    app()


if __name__ == "__main__":
    main()
