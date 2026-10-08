#!/usr/bin/env python3
"""合并公式相似度、训练统计和 noise/Wasserstein 测试指标。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Sequence
import pandas as pd
from paths import LOGS_ROOT

SIMILARITY_SHEET = "Merged"
TRAINED_SHEET = "Top_Programs"
NOISE_SHEET = 0

SIMILARITY_COLUMNS = ["ID", "GenerationFormula", "formula", "SimilarityScore", "SimilarityLevel"]
TRAINED_COLUMNS = ["problem_id", "score", "complexity", "mean_nmse", "min_nmse", "max_nmse", "median_nmse", "std_nmse"]
TRAINED_RENAME = {
    "mean_nmse": "trained_mean_nmse",
    "min_nmse": "trained_min_nmse",
    "max_nmse": "trained_max_nmse",
    "median_nmse": "trained_median_nmse",
    "std_nmse": "trained_std_nmse",
}
NOISE_EXCLUDE = {"problem_id", "top_rank", "function", "formula", "independent_variables", "target", "parameter_count"}
DEFAULT_OUTPUT = "merged_statistics.xlsx"


class MergeError(ValueError):
    pass


def normalize_id(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lstrip("\ufeff").strip()
    if re.fullmatch(r"[Pp]\d+", text):
        return text.upper()
    return text


def require_columns(df: pd.DataFrame, cols: Sequence[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MergeError(f"{name} 缺少字段 {missing}；现有字段：{list(df.columns)}")


def read_excel(path: Path, sheet) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{path}")
    return pd.read_excel(path, sheet_name=sheet)


def prepare_unique(df: pd.DataFrame, id_col: str, name: str) -> pd.DataFrame:
    df = df.copy()
    df["_merge_id"] = df[id_col].map(normalize_id)
    if (df["_merge_id"] == "").any():
        raise MergeError(f"{name} 存在空 {id_col}")
    dup = df.loc[df["_merge_id"].duplicated(keep=False), "_merge_id"].unique().tolist()
    if dup:
        raise MergeError(f"{name} 的 ID 不唯一：{', '.join(map(str, dup))}")
    return df

def main() -> int:
    parser = argparse.ArgumentParser(description="合并相似度表、训练 Top1 表和 noise/Wasserstein 测试拟合表")
    parser.add_argument("--similarity", type=Path, default=LOGS_ROOT / "generation_formula_and_top1_formula_logs_similarity_evaluated.xlsx")
    parser.add_argument("--trained", type=Path, default=LOGS_ROOT / "all_problems_checkpoint_top1.xlsx")
    parser.add_argument("--noise-fit", type=Path, default=LOGS_ROOT / "all_problems_checkpoint_top1_noise_subdirs_fit.xlsx")
    parser.add_argument("--output", type=Path, default=None, help="输出文件；默认放在 similarity 文件所在目录")
    args = parser.parse_args()

    similarity_path = args.similarity.expanduser().resolve()
    trained_path = args.trained.expanduser().resolve()
    noise_path = args.noise_fit.expanduser().resolve()
    output_path = args.output.expanduser().resolve() if args.output else similarity_path.parent / DEFAULT_OUTPUT

    sim = read_excel(similarity_path, SIMILARITY_SHEET)
    trained = read_excel(trained_path, TRAINED_SHEET)
    noise = read_excel(noise_path, NOISE_SHEET)

    require_columns(sim, SIMILARITY_COLUMNS, "Similarity 表")
    require_columns(trained, TRAINED_COLUMNS, "Trained Top1 表")
    require_columns(noise, ["problem_id"], "Noise-fit 表")

    if "top_rank" in trained.columns:
        trained = trained.loc[trained["top_rank"].eq(1)].copy()
    if "top_rank" in noise.columns:
        noise = noise.loc[noise["top_rank"].eq(1)].copy()

    sim = prepare_unique(sim[SIMILARITY_COLUMNS], "ID", "Similarity 表")
    trained = prepare_unique(trained[TRAINED_COLUMNS], "problem_id", "Trained Top1 表")

    noise_keep = [c for c in noise.columns if c not in NOISE_EXCLUDE]
    noise = prepare_unique(noise[["problem_id"] + noise_keep], "problem_id", "Noise-fit 表")

    trained_data = trained[["_merge_id"] + TRAINED_COLUMNS[1:]].rename(columns=TRAINED_RENAME)
    noise_data = noise[["_merge_id"] + noise_keep]

    result = sim.merge(trained_data, on="_merge_id", how="left", validate="one_to_one", sort=False)
    result = result.merge(noise_data, on="_merge_id", how="left", validate="one_to_one", sort=False)

    missing_trained = result.loc[result["score"].isna(), "ID"].tolist()
    first_noise_col = noise_keep[0] if noise_keep else None
    missing_noise = result.loc[result[first_noise_col].isna(), "ID"].tolist() if first_noise_col else result["ID"].tolist()
    if missing_trained:
        raise MergeError("Trained Top1 表中缺少 ID：" + ", ".join(missing_trained))
    if missing_noise:
        raise MergeError("Noise-fit 表中缺少 ID：" + ", ".join(missing_noise))

    result = result.drop(columns="_merge_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, sheet_name="Merged", index=False)

    print(f"完成：{len(result)} 行，{len(result.columns)} 列")
    print("输出字段：")
    print("  " + " | ".join(result.columns))
    print(f"输出文件：{output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, MergeError, ValueError) as exc:
        print(f"Error: {exc}")
        raise SystemExit(2)
