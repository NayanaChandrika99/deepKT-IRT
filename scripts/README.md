# ABOUTME: Documents utility and CLI scripts housed in scripts/.
# ABOUTME: Explains how demo Trace CLI interacts with model artifacts.

## Scripts

- `demo_trace.py`: Typer CLI that joins student mastery and item health to narrate a learning trace for a given student, topic, and time window. The script currently emits placeholder text but keeps the interface stable for future automation.
- Add dataset download helpers or preprocessing utilities here and reference them inside the `Makefile data` target.

## Usage

```
python scripts/demo_trace.py --student-id 123 --topic fractions --time-window 2023-W15
```

Expected output (today): structured TODO with the resolved config paths. After the models exist, this command should read `reports/student_state.parquet` and `reports/item_params.parquet`, filter by keys, and render a short recommendation narrative.
