"""
split_csv_tool.py
-----------------
Tool for splitting a multi-column CSV into individual single-column CSV files.
Each column is saved as `{column_header}.csv` (header-less) in the target directory.

Usage (as an agent tool):
    result = split_csv_tool(csv_path="data/sensor_log.csv", output_dir="data/split/")
    # result is a JSON string — pass as JSON to the LLM
"""

import os
import csv
import json
from pathlib import Path
from typing import Optional
from basic_tools import workspace
workspace      = Path(__file__).parent / "workspace"

def _resolve_path(path: str) -> str:
    """
    Resolve path relative to workspace, matching fft_tool.py convention.
    If path is absolute, use as-is; otherwise, treat as relative to workspace.
    """
    if os.path.isabs(path):
        return path
    return str(workspace / path)

def split_csv_tool(
    csv_path:      str,
    output_dir:    str,
    columns:       Optional[list[str]] = None,  # None = all columns
    skip_rows:     int                  = 0,     # rows to skip at top (e.g. 1 for meta header)
) -> str:
    """
    Split a multi-column CSV into individual single-column CSV files.

    Parameters
    ----------
    csv_path   : path to the source CSV file
    output_dir : directory where split column CSVs will be saved
    columns    : optional list of column names to extract.
                 If None, all columns are extracted.
    skip_rows  : number of rows to skip at the top of the file (default 0).
                 Set to 1 if the first row is a metadata/comment row.

    Returns
    -------
    JSON string with:
      - source_file     : original file path
      - output_dir      : where files were saved
      - skipped_columns : columns in the file that were NOT extracted
      - saved_files     : list of {filename, column_name, row_count} dicts
      - error           : error message if something went wrong (null if OK)
    """
    abs_csv_path = _resolve_path(csv_path)
    abs_output_dir = _resolve_path(output_dir)
    
    result = {
        "source_file":     csv_path,
        "output_dir":      output_dir,
        "skipped_columns": [],
        "saved_files":     [],
        "error":           None,
    }

    # ── Validate source file ─────────────────────────────────────────────────
    if not os.path.isfile(abs_csv_path):
        result["error"] = f"File not found: {abs_csv_path}"
        return json.dumps(result, indent=2)

    # ── Ensure output dir exists ─────────────────────────────────────────────
    os.makedirs(abs_output_dir, exist_ok=True)

    # ── Read CSV ─────────────────────────────────────────────────────────────
    try:
        with open(abs_csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        result["error"] = f"Failed to read CSV: {e}"
        return json.dumps(result, indent=2)

    if len(rows) < 2:
        result["error"] = f"CSV has fewer than 2 rows — nothing to split."
        return json.dumps(result, indent=2)

    # ── Handle skip_rows ──────────────────────────────────────────────────────
    if skip_rows >= len(rows):
        result["error"] = f"skip_rows={skip_rows} exceeds file length ({len(rows)} rows)."
        return json.dumps(result, indent=2)

    header_row = rows[skip_rows]
    data_rows  = rows[skip_rows + 1:]

    if not header_row:
        result["error"] = "CSV header row is empty."
        return json.dumps(result, indent=2)

    col_count = len(header_row)

    # ── Determine which columns to process ───────────────────────────────────
    if columns is None:
        columns_to_process = header_row
    else:
        missing = [c for c in columns if c not in header_row]
        if missing:
            result["error"] = f"Requested columns not found in CSV: {missing}"
            return json.dumps(result, indent=2)
        columns_to_process = columns
        result["skipped_columns"] = [c for c in header_row if c not in columns]

    # ── Build column index map ────────────────────────────────────────────────
    col_idx = {name: i for i, name in enumerate(header_row)}

    # ── Write one file per column ─────────────────────────────────────────────
    for col_name in columns_to_process:
        idx = col_idx[col_name]

        # Sanitise filename — remove/replace unsafe characters
        safe_name = _sanitise_filename(col_name)
        out_path = os.path.join(abs_output_dir, f"{safe_name}.csv")

        try:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for row in data_rows:
                    # Guard against rows shorter than expected
                    value = row[idx] if idx < len(row) else ""
                    writer.writerow([value])
        except Exception as e:
            result["error"] = f"Failed to write column '{col_name}': {e}"
            return json.dumps(result, indent=2)

        result["saved_files"].append({
            "filename":   f"{safe_name}.csv",
            "path":       os.path.join(output_dir, f"{safe_name}.csv"),  
            "column_name": col_name,
            "row_count":  len(data_rows),
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    result["summary"] = (
        f"Split {len(result['saved_files'])} column(s) from '{csv_path}' → "
        f"'{output_dir}'. "
        f"Sample rate must be provided by user when running fft_tool."
    )

    return json.dumps(result, indent=2)


def _sanitise_filename(name: str) -> str:
    """
    Turn a column header into a safe filename.
    Replaces spaces and unsafe chars with underscores, strips trailing underscores.
    """
    # Replace common separators with underscore
    safe = name.strip()
    for ch in (" ", "-", "/", "\\", ":", "*", "?", '"', "<", ">", "|"):
        safe = safe.replace(ch, "_")
    # Collapse multiple underscores
    while "__" in safe:
        safe = safe.replace("__", "_")
    safe = safe.strip("_")
    return safe if safe else "unnamed_column"


# =============================================================================
# Smoke test
# =============================================================================

if __name__ == "__main__":
    import tempfile

    # Create workspace directory for testing
    workspace = Path(__file__).parent / "workspace"
    workspace.mkdir(exist_ok=True)

    # ── Toy CSV ───────────────────────────────────────────────────────────────
    csv_content = (
        "timestamp,accel_x,accel_y,accel_z,temperature\n"
        "0.000,0.012,0.034,9.81,22.5\n"
        "0.001,0.015,0.031,9.80,22.6\n"
        "0.002,0.018,0.028,9.79,22.7\n"
        "0.003,0.021,0.025,9.78,22.8\n"
    )

    # Save test CSV in workspace
    test_csv = workspace / "test_data.csv"
    with open(test_csv, "w") as f:
        f.write(csv_content)

    # Test split (using relative paths)
    print(f"Source : {test_csv}")
    print(f"Workspace: {workspace}")
    print()

    out = split_csv_tool(
        csv_path="test_data.csv",  # relative to workspace
        output_dir="split",         # relative to workspace
        skip_rows=0
    )
    print(out)

    # ── Show what was written ─────────────────────────────────────────────────
    print("\n── Files created ──────────────────────────────────────")
    split_dir = workspace / "split"
    for fname in sorted(os.listdir(split_dir)):
        fpath = split_dir / fname
        with open(fpath) as f:
            lines = f.readlines()
        print(f"  {fname}  ({len(lines)} rows)")
        for line in lines[:3]:
            print(f"    {line.rstrip()}")
        if len(lines) > 3:
            print(f"    ... ({len(lines)-3} more rows)")