#!/usr/bin/env python3
"""按 Range.ID == Top1.problem_id 合并两个表格。

输出包含：ID、GenerationFormula、Target、IndependentVars、ParameterRange、FixedConstantValues、formula。
默认输出到 Top1 文件所在目录，不覆盖输入文件。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Sequence
from paths import BENCHMARK_FILE, LOGS_ROOT

try:
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少运行依赖。请先执行：pip install pandas openpyxl"
    ) from exc


# ======================== 常用参数：通常只需修改这里 ========================
RANGE_FILE = BENCHMARK_FILE
TOP1_FILE = LOGS_ROOT / "all_problems_checkpoint_top1.xlsx"

RANGE_SHEET: str | int = "Sampling design"
TOP1_SHEET: str | int = "Top_Programs"

RANGE_ID_COLUMN = "ID"
GENERATION_FORMULA_COLUMN = "GenerationFormula"
TARGET_COLUMN = "Target"
INDEPENDENT_VARS_COLUMN = "IndependentVars"
PARAMETER_RANGE_COLUMN = "ParameterRange"
FIXED_CONSTANT_VALUES_COLUMN = "FixedConstantValues"
TOP1_ID_COLUMN = "problem_id"
TOP1_FORMULA_COLUMN = "formula"

OUTPUT_FILENAME = "generation_formula_and_top1_formula.xlsx"
# ===========================================================================


class MergeTableError(ValueError):
    """输入表结构或 ID 对应关系不符合要求。"""


def normalize_sheet(value: str | int) -> str | int:
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def read_table(path: Path, sheet: str | int) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=normalize_sheet(sheet))
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise MergeTableError(f"不支持的文件格式：{path.suffix}")


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    table_name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise MergeTableError(
            f"{table_name} 缺少列 {missing}；现有列：{list(frame.columns)}"
        )


def normalize_id(value: Any) -> str:
    """清理空格、BOM，并统一 P01/p01 等大小写。"""
    if pd.isna(value):
        return ""
    text = str(value).lstrip("\ufeff").strip()
    if re.fullmatch(r"[Pp]\d+", text):
        return text.upper()
    return text


def find_duplicates(values: pd.Series) -> list[str]:
    duplicated = values[values.duplicated(keep=False) & values.ne("")]
    return sorted(duplicated.unique().tolist())


def merge_tables(
    range_frame: pd.DataFrame,
    top1_frame: pd.DataFrame,
    range_id_column: str,
    generation_formula_column: str,
    target_column: str,
    independent_vars_column: str,
    parameter_range_column: str,
    fixed_constant_values_column: str,
    top1_id_column: str,
    top1_formula_column: str,
) -> pd.DataFrame:
    require_columns(
        range_frame,
        [range_id_column, generation_formula_column, target_column, independent_vars_column, parameter_range_column, fixed_constant_values_column],
        "Range 表",
    )
    require_columns(
        top1_frame,
        [top1_id_column, top1_formula_column],
        "Top1 表",
    )

    range_columns = [range_id_column, generation_formula_column, target_column, independent_vars_column, parameter_range_column, fixed_constant_values_column]
    range_data = range_frame[range_columns].copy()
    top1_data = top1_frame[[top1_id_column, top1_formula_column]].copy()
    range_data["_merge_id"] = range_data[range_id_column].map(normalize_id)
    top1_data["_merge_id"] = top1_data[top1_id_column].map(normalize_id)

    empty_range = range_data.index[range_data["_merge_id"].eq("")].tolist()
    empty_top1 = top1_data.index[top1_data["_merge_id"].eq("")].tolist()
    if empty_range:
        raise MergeTableError(
            f"Range 表存在空 ID，Excel 数据行：{[index + 2 for index in empty_range]}"
        )
    if empty_top1:
        raise MergeTableError(
            f"Top1 表存在空 problem_id，Excel 数据行："
            f"{[index + 2 for index in empty_top1]}"
        )

    duplicate_range_ids = find_duplicates(range_data["_merge_id"])
    if duplicate_range_ids:
        raise MergeTableError(
            "Range 表的 ID 不唯一，无法确定 GenerationFormula："
            + ", ".join(duplicate_range_ids)
        )

    merged = top1_data.merge(
        range_data[["_merge_id", generation_formula_column, target_column, independent_vars_column, parameter_range_column, fixed_constant_values_column]],
        on="_merge_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    missing_ids = merged.loc[
        merged[generation_formula_column].isna(), "_merge_id"
    ].drop_duplicates().tolist()
    if missing_ids:
        raise MergeTableError(
            "以下 Top1 problem_id 在 Range 表中找不到：" + ", ".join(missing_ids)
        )

    result = merged[["_merge_id", generation_formula_column, target_column, independent_vars_column, parameter_range_column, fixed_constant_values_column, top1_formula_column]].copy()
    result.columns = ["ID", "GenerationFormula", "Target", "IndependentVars", "ParameterRange", "FixedConstantValues", "formula"]
    return result


def write_output(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".xlsx":
        frame.to_excel(output_path, sheet_name="Merged", index=False, engine="openpyxl")
    elif suffix == ".csv":
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        raise MergeTableError("输出文件必须是 .xlsx 或 .csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按 Range.ID == Top1.problem_id 合并 Range 字段和 Top1 formula"
        )
    )
    parser.add_argument("--range-file", type=Path, default=RANGE_FILE)
    parser.add_argument("--top1-file", type=Path, default=TOP1_FILE)
    parser.add_argument("--range-sheet", default=RANGE_SHEET)
    parser.add_argument("--top1-sheet", default=TOP1_SHEET)
    parser.add_argument("--range-id-column", default=RANGE_ID_COLUMN)
    parser.add_argument(
        "--generation-formula-column",
        default=GENERATION_FORMULA_COLUMN,
    )
    parser.add_argument("--target-column", default=TARGET_COLUMN)
    parser.add_argument("--independent-vars-column", default=INDEPENDENT_VARS_COLUMN)
    parser.add_argument("--parameter-range-column", default=PARAMETER_RANGE_COLUMN)
    parser.add_argument("--fixed-constant-values-column", default=FIXED_CONSTANT_VALUES_COLUMN)
    parser.add_argument("--top1-id-column", default=TOP1_ID_COLUMN)
    parser.add_argument("--top1-formula-column", default=TOP1_FORMULA_COLUMN)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "输出路径；省略时自动输出到 Top1 文件目录下的 "
            f"{OUTPUT_FILENAME}"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    range_path = args.range_file.expanduser().resolve()
    top1_path = args.top1_file.expanduser().resolve()

    range_frame = read_table(range_path, args.range_sheet)
    top1_frame = read_table(top1_path, args.top1_sheet)
    result = merge_tables(
        range_frame,
        top1_frame,
        args.range_id_column,
        args.generation_formula_column,
        args.target_column,
        args.independent_vars_column,
        args.parameter_range_column,
        args.fixed_constant_values_column,
        args.top1_id_column,
        args.top1_formula_column,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else top1_path.parent / OUTPUT_FILENAME
    )
    if output_path in {range_path, top1_path}:
        raise MergeTableError("输出文件不能覆盖任一输入文件")
    write_output(result, output_path)
    print(f"完成：合并 {len(result)} 行")
    print(f"输出文件：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, MergeTableError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)
