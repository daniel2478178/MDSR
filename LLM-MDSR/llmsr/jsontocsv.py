from pathlib import Path
import json
import pandas as pd


# ============================================================
# Configuration
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1] / "logs" / "p20" / "samples"
OUTPUT_FILE = ROOT_DIR / "LLMSR_results_summary.xlsx"


# ============================================================
# Read one JSON
# ============================================================

def read_one_json(json_file: Path, root_dir: Path) -> dict:
    """Read one LLMSR result JSON."""

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    relative_path = json_file.relative_to(root_dir)
    dataset_id = relative_path.parts[0] if len(relative_path.parts) >= 2 else json_file.parent.name

    sample_order = data.get("sample_order")
    function = data.get("function", "")
    score = data.get("score")

    nmse = None
    if score is not None:
        try:
            score = float(score)
            nmse = -score
        except (TypeError, ValueError):
            pass

    return {
        "ID": dataset_id,
        "json_file": json_file.name,
        "sample_order": sample_order,
        "function": function,
        "score": score,
        "NMSE": nmse,
        "relative_path": str(relative_path),
    }


# ============================================================
# Collect all JSON files
# ============================================================

def collect_results(root_dir: Path) -> pd.DataFrame:
    json_files = sorted(root_dir.rglob("*.json"))

    print("=" * 80)
    print("LLMSR JSON Result Collector")
    print("=" * 80)
    print(f"Root directory : {root_dir}")
    print(f"JSON files     : {len(json_files)}")
    print("=" * 80)

    rows = []

    for i, json_file in enumerate(json_files, start=1):
        print(f"[{i:5d}/{len(json_files):5d}] {json_file}")

        try:
            row = read_one_json(json_file, root_dir)
            row["status"] = "OK"
            row["error"] = ""

        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}")

            relative_path = json_file.relative_to(root_dir)
            dataset_id = relative_path.parts[0] if len(relative_path.parts) >= 2 else json_file.parent.name

            row = {
                "ID": dataset_id,
                "json_file": json_file.name,
                "sample_order": None,
                "function": "",
                "score": None,
                "NMSE": None,
                "relative_path": str(relative_path),
                "status": "FAILED",
                "error": f"{type(e).__name__}: {e}",
            }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def main():
    if not ROOT_DIR.exists():
        raise FileNotFoundError(f"ROOT_DIR does not exist:\n{ROOT_DIR}")

    df = collect_results(ROOT_DIR)

    if df.empty:
        print("No JSON files found.")
        return

    df = df.sort_values(
        by=["ID", "sample_order"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)

    columns = [
        "ID",
        "sample_order",
        "score",
        "NMSE",
        "function",
        "json_file",
        "relative_path",
        "status",
        "error",
    ]
    df = df[[col for col in columns if col in df.columns]]

    df.to_excel(OUTPUT_FILE, index=False, engine="openpyxl")

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)
    print(f"Total JSON files : {len(df)}")
    print(f"Successful       : {(df['status'] == 'OK').sum()}")
    print(f"Failed           : {(df['status'] == 'FAILED').sum()}")
    print(f"Datasets         : {df['ID'].nunique()}")
    print(f"Output           : {OUTPUT_FILE}")
    print("=" * 80)


if __name__ == "__main__":
    main()
