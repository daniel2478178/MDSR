#!/usr/bin/env python3
"""读取 Excel 的 function，拟合多个数据子目录中的多组 CSV，并汇总指标。

数据列识别规则：
1. 从 function 的函数签名读取独立变量（排除 params）；
2. 在 CSV 中查找这些同名列；
3. 除独立变量外若只剩一列，则自动将其作为 target；
4. 也可用 --target-column 显式指定统一的 target 列名。

默认数据路径为：<DATA_ROOT>/<subdir>/<problem_id>/0.csv ... 7.csv。
如文件布局不同，可修改 DATA_PATH_TEMPLATE 或使用 --data-template。
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import multiprocessing
import os
import re
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from paths import LOGS_ROOT, TEST_DATA_ROOT

try:
    import numpy as np
    import pandas as pd
    from scipy.optimize import least_squares
except ModuleNotFoundError as exc:
    raise SystemExit(
        "缺少运行依赖。请先执行：pip install numpy pandas scipy openpyxl"
    ) from exc


# ======================== 常用参数：通常只需修改这里 ========================
XLSX_FILE = LOGS_ROOT / "all_problems_checkpoint_top1.xlsx"
XLSX_SHEET = "Top_Programs"
NOISE_DATA_ROOT = TEST_DATA_ROOT
DATA_SUBDIRS = ("w0",)  #("0", "001", "003", "005", "01","w0","w025","w04")
DATASET_INDICES = tuple(range(8))
DATA_PATH_TEMPLATE = "{subdir}/{problem_id}/{dataset_index}.csv"
OUTPUT_FILE: Path | None = None  # None：自动保存在 Excel 旁边

ID_COLUMN = "problem_id"
FUNCTION_COLUMN = "function"
TARGET_COLUMN: str | None = None  # None：自动识别 target
METRIC_PREFIX = "noise"

TOP_RANK: int | None = None  # None：处理表中所有行；也可设为 1
WORKERS = min(4, os.cpu_count() or 1)
MAX_NFEV = 10_000
RESTARTS = 3
RANDOM_SEED = 20260824
FAILURE_VALUE = 1.0e100
# ===========================================================================


ALLOWED_NUMPY_FUNCTIONS = {
    "sqrt",
    "exp",
    "log",
    "sin",
    "cos",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "arcsin",
    "arccos",
    "arctan",
    "asin",
    "acos",
    "atan",
    "arcsinh",
    "arccosh",
    "arctanh",
    "abs",
    "fabs",
    "power",
    "square",
    "maximum",
    "minimum",
}


class FitDataError(ValueError):
    """输入表、函数或数据文件不符合要求。"""


@dataclass(frozen=True)
class PreparedFunction:
    source: str
    function_name: str
    argument_names: tuple[str, ...]
    variable_names: tuple[str, ...]
    parameter_count: int
    callable_object: Any


@dataclass(frozen=True)
class FitResult:
    row_position: int
    problem_id: str
    top_rank: Any
    data_subdir: str
    data_files: tuple[str, ...]
    variable_names: tuple[str, ...]
    target_name: str
    sample_counts: tuple[int, ...]
    parameter_count: int
    fitted_parameters: tuple[tuple[float, ...], ...]
    dataset_metrics: tuple[tuple[float, float, float], ...]
    mse: float
    nmse: float
    r2: float
    status: str
    error: str


def protected_power(base: Any, exponent: Any) -> Any:
    """优化过程中尽量保持幂运算为实数，减少试探参数导致的 NaN。"""
    base_array = np.asarray(base)
    exponent_array = np.asarray(exponent)
    with np.errstate(all="ignore"):
        magnitude = np.power(np.abs(base_array), exponent_array)
        nearest_integer = np.rint(exponent_array)
        is_integer = np.isclose(
            exponent_array, nearest_integer, rtol=0.0, atol=1.0e-10
        )
        integer_sign = np.where(np.mod(nearest_integer, 2.0) == 0.0, 1.0, -1.0)

        nearest_third_numerator = np.rint(exponent_array * 3.0)
        nearest_third = nearest_third_numerator / 3.0
        is_third = np.isclose(
            exponent_array, nearest_third, rtol=0.0, atol=1.0e-10
        )
        third_sign = np.where(
            np.mod(nearest_third_numerator, 2.0) == 0.0, 1.0, -1.0
        )
        negative_sign = np.where(
            is_integer, integer_sign, np.where(is_third, third_sign, 1.0)
        )
        return np.where(
            base_array < 0.0,
            negative_sign * magnitude,
            magnitude,
        )


class ProtectedPowerTransformer(ast.NodeTransformer):
    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Pow):
            return ast.copy_location(
                ast.Call(
                    func=ast.Name(id="_protected_power", ctx=ast.Load()),
                    args=[node.left, node.right],
                    keywords=[],
                ),
                node,
            )
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr == "power"
        ):
            node.func = ast.Name(id="_protected_power", ctx=ast.Load())
        return node


class FunctionValidator(ast.NodeVisitor):
    """只允许拟合公式所需的简单 NumPy 函数代码。"""

    BINARY_OPERATORS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod)
    UNARY_OPERATORS = (ast.UAdd, ast.USub)

    def __init__(self, argument_names: Sequence[str]) -> None:
        self.allowed_names = set(argument_names) | {"np"}

    def validate_statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.Expr):
            if not (
                isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            ):
                raise FitDataError("函数中只允许字符串说明、赋值和 return")
            return
        if isinstance(statement, ast.Assign):
            self.visit(statement.value)
            for target in statement.targets:
                if not isinstance(target, ast.Name):
                    raise FitDataError("只允许给普通局部变量赋值")
                self.allowed_names.add(target.id)
            return
        if isinstance(statement, ast.AnnAssign) and statement.value is not None:
            self.visit(statement.value)
            if not isinstance(statement.target, ast.Name):
                raise FitDataError("只允许给普通局部变量赋值")
            self.allowed_names.add(statement.target.id)
            return
        if isinstance(statement, ast.Return):
            if statement.value is None:
                raise FitDataError("return 没有表达式")
            self.visit(statement.value)
            return
        raise FitDataError(f"函数中不支持语句：{type(statement).__name__}")

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, self.BINARY_OPERATORS):
            raise FitDataError(f"不支持运算符：{type(node.op).__name__}")
        self.visit(node.left)
        self.visit(node.right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, self.UNARY_OPERATORS):
            raise FitDataError(f"不支持一元运算：{type(node.op).__name__}")
        self.visit(node.operand)

    def visit_Call(self, node: ast.Call) -> None:
        if node.keywords:
            raise FitDataError("函数调用中不允许关键字参数")
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "np"
            and node.func.attr in ALLOWED_NUMPY_FUNCTIONS
        ):
            raise FitDataError(f"不支持函数调用：{ast.unparse(node.func)}")
        for argument in node.args:
            self.visit(argument)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.allowed_names:
            raise FitDataError(f"函数使用了未定义名称：{node.id}")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if not (
            isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr in ALLOWED_NUMPY_FUNCTIONS | {"pi", "e"}
        ):
            raise FitDataError(f"不支持属性：{ast.unparse(node)}")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if not (
            isinstance(node.value, ast.Name)
            and node.value.id == "params"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
            and node.slice.value >= 0
        ):
            raise FitDataError("只支持 params[非负整数] 形式的参数")

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise FitDataError("公式中只允许数值常量")

    def generic_visit(self, node: ast.AST) -> None:
        raise FitDataError(f"公式包含不支持的语法：{type(node).__name__}")


def normalize_function_source(source: Any) -> str:
    text = str(source or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:python)?\s*", "", text, count=1, flags=re.I)
        text = re.sub(r"\s*```$", "", text, count=1)
    return text.strip()


def prepare_function(source: Any) -> PreparedFunction:
    source_text = normalize_function_source(source)
    if not source_text or source_text.lower() == "nan":
        raise FitDataError("function 为空")
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        raise FitDataError(f"function 语法错误：{exc}") from exc

    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(functions) != 1 or len(tree.body) != 1:
        raise FitDataError("function 必须只包含一个函数定义")
    function_node = functions[0]
    if function_node.decorator_list:
        raise FitDataError("function 不允许装饰器")
    if function_node.args.vararg or function_node.args.kwarg:
        raise FitDataError("function 不允许 *args 或 **kwargs")

    argument_names = tuple(argument.arg for argument in function_node.args.args)
    if "params" not in argument_names:
        raise FitDataError("function 的参数列表中没有 params")
    variable_names = tuple(name for name in argument_names if name != "params")
    if not variable_names:
        raise FitDataError("function 中没有独立变量")

    validator = FunctionValidator(argument_names)
    for statement in function_node.body:
        validator.validate_statement(statement)

    parameter_indices = [
        node.slice.value
        for node in ast.walk(function_node)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "params"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, int)
    ]
    parameter_count = max(parameter_indices, default=-1) + 1

    protected_tree = ProtectedPowerTransformer().visit(copy.deepcopy(tree))
    ast.fix_missing_locations(protected_tree)
    environment: dict[str, Any] = {
        "np": np,
        "_protected_power": protected_power,
        "__builtins__": {},
    }
    exec(compile(protected_tree, "<function_from_excel>", "exec"), environment)
    callable_object = environment.get(function_node.name)
    if not callable(callable_object):
        raise FitDataError("无法建立 function 可调用对象")

    return PreparedFunction(
        source=source_text,
        function_name=function_node.name,
        argument_names=argument_names,
        variable_names=variable_names,
        parameter_count=parameter_count,
        callable_object=callable_object,
    )


def normalize_column_names(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.columns = [str(column).lstrip("\ufeff").strip() for column in result.columns]
    if len(result.columns) != len(set(result.columns)):
        raise FitDataError("清理 BOM 和空格后，CSV 出现重复列名")
    return result


def detect_target_column(
    columns: Sequence[str],
    variable_names: Sequence[str],
    explicit_target: str | None,
) -> str:
    if explicit_target:
        if explicit_target not in columns:
            raise FitDataError(
                f"指定 target 列 {explicit_target!r} 不存在；现有列：{list(columns)}"
            )
        if explicit_target in variable_names:
            raise FitDataError("target 列不能同时是独立变量")
        return explicit_target

    candidates = [column for column in columns if column not in variable_names]
    if len(candidates) == 1:
        return candidates[0]

    preferred = [
        column
        for column in candidates
        if column.lower() in {"target", "output", "y"}
    ]
    if len(preferred) == 1:
        return preferred[0]
    raise FitDataError(
        "无法唯一确定 target。独立变量为 "
        f"{list(variable_names)}，其余列为 {candidates}；请使用 --target-column 指定"
    )


def read_noise_dataset(
    path: Path,
    variable_names: Sequence[str],
    target_column: str | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, str, int]:
    if not path.is_file():
        raise FileNotFoundError(f"噪声数据不存在：{path}")
    frame = normalize_column_names(pd.read_csv(path, encoding="utf-8-sig"))
    missing = [name for name in variable_names if name not in frame.columns]
    if missing:
        raise FitDataError(
            f"{path} 缺少独立变量列 {missing}；现有列：{list(frame.columns)}"
        )
    target = detect_target_column(list(frame.columns), variable_names, target_column)
    required = [*variable_names, target]
    numeric = frame[required].apply(pd.to_numeric, errors="coerce")
    valid = np.all(np.isfinite(numeric.to_numpy(dtype=float)), axis=1)
    numeric = numeric.loc[valid]
    if numeric.empty:
        raise FitDataError(f"{path} 没有完整的有限数值行")
    x_columns = {
        name: numeric[name].to_numpy(dtype=float) for name in variable_names
    }
    y = numeric[target].to_numpy(dtype=float)
    return x_columns, y, target, len(numeric)


def evaluate_function(
    prepared: PreparedFunction,
    x_columns: Mapping[str, np.ndarray],
    parameters: Sequence[float],
) -> np.ndarray:
    call_arguments = [
        np.asarray(parameters, dtype=float)
        if name == "params"
        else x_columns[name]
        for name in prepared.argument_names
    ]
    with np.errstate(all="ignore"):
        prediction = prepared.callable_object(*call_arguments)
    sample_count = len(next(iter(x_columns.values())))
    prediction = np.asarray(prediction)
    if prediction.ndim == 0:
        prediction = np.full(sample_count, prediction.item())
    else:
        try:
            prediction = np.broadcast_to(prediction, (sample_count,))
        except ValueError as exc:
            raise FitDataError("function 输出无法转换成一维预测值") from exc
    if np.iscomplexobj(prediction):
        prediction = np.where(
            np.abs(prediction.imag) <= 1.0e-12,
            prediction.real,
            np.nan,
        )
    try:
        return prediction.astype(float, copy=False)
    except (TypeError, ValueError) as exc:
        raise FitDataError("function 输出不是实数") from exc


def make_initial_candidates(
    parameter_count: int,
    restarts: int,
    seed: int,
) -> list[np.ndarray]:
    initial = np.ones(parameter_count, dtype=float)
    candidates = [initial]
    rng = np.random.default_rng(seed)
    for _ in range(1, restarts):
        candidates.append(initial + rng.normal(0.0, 0.5, size=parameter_count))
    return candidates


def fit_parameters(
    prepared: PreparedFunction,
    x_columns: Mapping[str, np.ndarray],
    y: np.ndarray,
    max_nfev: int,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if prepared.parameter_count == 0:
        prediction = evaluate_function(prepared, x_columns, ())
        if not np.all(np.isfinite(prediction)):
            raise FitDataError("无参数 function 产生非有限预测值")
        return np.empty(0, dtype=float), prediction

    penalty = np.maximum(1.0, np.abs(y)) * 1.0e6
    best_parameters: np.ndarray | None = None
    best_sse = math.inf

    def residual(parameters: np.ndarray) -> np.ndarray:
        nonlocal best_parameters, best_sse
        prediction_local = evaluate_function(prepared, x_columns, parameters)
        difference = prediction_local - y
        finite = np.isfinite(difference)
        if np.all(finite):
            sse = float(np.dot(difference, difference))
            if math.isfinite(sse) and sse < best_sse:
                best_sse = sse
                best_parameters = np.asarray(parameters, dtype=float).copy()
        return np.where(finite, difference, penalty)

    fitting_errors: list[str] = []
    for initial in make_initial_candidates(prepared.parameter_count, restarts, seed):
        residual(initial)
        try:
            result = least_squares(
                residual,
                initial,
                method="trf",
                x_scale="jac",
                max_nfev=max_nfev,
            )
            residual(result.x)
        except (ValueError, FloatingPointError) as exc:
            fitting_errors.append(str(exc))

    if best_parameters is None:
        details = " | ".join(fitting_errors[:3])
        raise FitDataError(
            "所有初值都无法产生有限预测值"
            + (f"；优化器信息：{details}" if details else "")
        )
    return best_parameters, evaluate_function(prepared, x_columns, best_parameters)


def calculate_metrics(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float, float]:
    residual = y - prediction
    sse = float(np.dot(residual, residual))
    if not math.isfinite(sse):
        raise FitDataError("残差平方和不是有限值")
    mse = sse / len(y)
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst == 0.0:
        if sse == 0.0:
            return mse, 0.0, 1.0
        raise FitDataError("target 方差为0，NMSE和R²无定义")
    nmse = sse / sst
    return mse, nmse, 1.0 - nmse


def format_data_path(
    data_root: Path,
    template: str,
    data_subdir: str,
    problem_id: str,
    top_rank: Any,
    dataset_index: int,
) -> Path:
    try:
        relative = template.format(
            subdir=data_subdir,
            data_subdir=data_subdir,
            problem_id=problem_id,
            rank=top_rank,
            top_rank=top_rank,
            dataset_index=dataset_index,
            filename=f"{dataset_index}.csv",
        )
    except KeyError as exc:
        raise FitDataError(f"data-template 包含未知占位符：{exc}") from exc
    path = Path(relative)
    return path if path.is_absolute() else data_root / path


def fit_row_subdir(
    row_position: int,
    row: Mapping[str, Any],
    data_root_text: str,
    data_template: str,
    data_subdir: str,
    subdir_position: int,
    dataset_indices: Sequence[int],
    id_column: str,
    function_column: str,
    target_column: str | None,
    max_nfev: int,
    restarts: int,
    seed: int,
    failure_value: float,
) -> FitResult:
    problem_id = str(row.get(id_column, "")).strip()
    top_rank = row.get("top_rank", row.get("TopRank", ""))
    data_paths = tuple(
        format_data_path(
            Path(data_root_text),
            data_template,
            data_subdir,
            problem_id,
            top_rank,
            dataset_index,
        )
        for dataset_index in dataset_indices
    )
    try:
        if not problem_id:
            raise FitDataError(f"第 {row_position + 2} 行的问题编号为空")
        prepared = prepare_function(row.get(function_column))
        targets: list[str] = []
        sample_counts: list[int] = []
        fitted_parameters: list[tuple[float, ...]] = []
        metrics: list[tuple[float, float, float]] = []
        failures: list[str] = []

        for dataset_index, data_path in zip(dataset_indices, data_paths):
            try:
                x_columns, y, detected_target, sample_count = read_noise_dataset(
                    data_path,
                    prepared.variable_names,
                    target_column,
                )
                parameters, prediction = fit_parameters(
                    prepared,
                    x_columns,
                    y,
                    max_nfev,
                    restarts,
                    seed
                    + row_position * 1009
                    + subdir_position * 100_003
                    + dataset_index * 37,
                )
                mse, nmse, r2 = calculate_metrics(y, prediction)
                targets.append(detected_target)
                sample_counts.append(sample_count)
                fitted_parameters.append(tuple(float(value) for value in parameters))
                metrics.append((mse, nmse, r2))
            except Exception as exc:
                failures.append(
                    f"{data_subdir}/{problem_id}/{dataset_index}.csv: "
                    f"{type(exc).__name__}: {exc}"
                )

        if failures:
            mean_mse = failure_value
            mean_nmse = failure_value
            mean_r2 = -failure_value
            status = "FAILED"
        else:
            if len(set(targets)) != 1:
                raise FitDataError(f"8组数据识别出不同 target：{targets}")
            mean_mse = float(np.mean([value[0] for value in metrics]))
            mean_nmse = float(np.mean([value[1] for value in metrics]))
            mean_r2 = float(np.mean([value[2] for value in metrics]))
            status = "OK"

        return FitResult(
            row_position=row_position,
            problem_id=problem_id,
            top_rank=top_rank,
            data_subdir=data_subdir,
            data_files=tuple(str(path) for path in data_paths),
            variable_names=prepared.variable_names,
            target_name=targets[0] if targets else "",
            sample_counts=tuple(sample_counts),
            parameter_count=prepared.parameter_count,
            fitted_parameters=tuple(fitted_parameters),
            dataset_metrics=tuple(metrics),
            mse=mean_mse,
            nmse=mean_nmse,
            r2=mean_r2,
            status=status,
            error=" | ".join(failures),
        )
    except Exception as exc:
        return FitResult(
            row_position=row_position,
            problem_id=problem_id or f"row-{row_position + 2}",
            top_rank=top_rank,
            data_subdir=data_subdir,
            data_files=tuple(str(path) for path in data_paths),
            variable_names=(),
            target_name="",
            sample_counts=(),
            parameter_count=0,
            fitted_parameters=(),
            dataset_metrics=(),
            mse=failure_value,
            nmse=failure_value,
            r2=-failure_value,
            status="FAILED",
            error=f"{type(exc).__name__}: {exc}",
        )


def read_table(path: Path, sheet: str | int) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        sheet_name: str | int = (
            int(sheet) if isinstance(sheet, str) and sheet.isdigit() else sheet
        )
        return pd.read_excel(path, sheet_name=sheet_name)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise FitDataError("输入表格必须是 .xlsx、.xlsm 或 .csv")


def choose_column(
    frame: pd.DataFrame,
    requested: str,
    alternatives: Sequence[str],
    label: str,
) -> str:
    if requested in frame.columns:
        return requested
    for candidate in alternatives:
        if candidate in frame.columns:
            return candidate
    raise FitDataError(
        f"找不到{label}列 {requested!r}；现有列：{list(frame.columns)}"
    )


def default_output_path(table_path: Path, metric_prefix: str) -> Path:
    return table_path.with_name(f"{table_path.stem}_{metric_prefix}_subdirs_fit.xlsx")


def write_output(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    elif output_path.suffix.lower() == ".xlsx":
        frame.to_excel(output_path, index=False, engine="openpyxl")
    else:
        raise FitDataError("输出文件必须是 .xlsx 或 .csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="用 Excel 的 function 拟合多个数据子目录中每个问题的0.csv到7.csv"
    )
    parser.add_argument("--xlsx", type=Path, default=XLSX_FILE, help="公式 Excel/CSV")
    parser.add_argument("--sheet", default=XLSX_SHEET, help="Excel 工作表名称或序号")
    parser.add_argument(
        "--data-root", type=Path, default=NOISE_DATA_ROOT, help="噪声数据根目录"
    )
    parser.add_argument(
        "--data-subdirs",
        nargs="+",
        default=list(DATA_SUBDIRS),
        help="数据子目录名称，可任意使用 001、w01、w05 等标签",
    )
    parser.add_argument(
        "--dataset-indices",
        nargs="+",
        type=int,
        default=list(DATASET_INDICES),
        help="每个问题拟合的数据编号；默认 0 1 2 3 4 5 6 7",
    )
    parser.add_argument(
        "--data-template",
        default=DATA_PATH_TEMPLATE,
        help=(
            "相对 data-root 的路径模板；支持 {subdir}、{problem_id}、"
            "{dataset_index}、{rank}、{filename}"
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE, help="输出 XLSX/CSV")
    parser.add_argument("--id-column", default=ID_COLUMN, help="问题编号列名")
    parser.add_argument("--function-column", default=FUNCTION_COLUMN, help="函数列名")
    parser.add_argument(
        "--target-column",
        default=TARGET_COLUMN,
        help="统一 target 列名；默认从每个 CSV 自动识别",
    )
    parser.add_argument("--metric-prefix", default=METRIC_PREFIX, help="指标列前缀")
    parser.add_argument("--top-rank", type=int, default=TOP_RANK, help="只拟合指定排名")
    parser.add_argument(
        "--problem-id", nargs="+", default=None, help="只拟合指定问题，如 P01 P02"
    )
    parser.add_argument("--workers", type=int, default=WORKERS, help="并行进程数")
    parser.add_argument("--max-nfev", type=int, default=MAX_NFEV)
    parser.add_argument("--restarts", type=int, default=RESTARTS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--failure-value", type=float, default=FAILURE_VALUE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers <= 0 or args.max_nfev <= 0 or args.restarts <= 0:
        raise FitDataError("workers、max-nfev 和 restarts 必须大于0")
    if not math.isfinite(args.failure_value) or args.failure_value <= 0:
        raise FitDataError("failure-value 必须是有限正数")

    table_path = args.xlsx.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    if not table_path.is_file():
        raise FileNotFoundError(f"公式表格不存在：{table_path}")
    if not data_root.is_dir():
        raise NotADirectoryError(f"噪声数据根目录不存在：{data_root}")

    frame = read_table(table_path, args.sheet)
    id_column = choose_column(frame, args.id_column, ("problem_id", "ID"), "问题编号")
    function_column = choose_column(
        frame, args.function_column, ("function", "program.body"), "function"
    )
    positions = list(range(len(frame)))
    if args.top_rank is not None:
        rank_column = choose_column(frame, "top_rank", ("TopRank", "TOPK"), "排名")
        ranks = pd.to_numeric(frame[rank_column], errors="coerce")
        positions = [position for position in positions if ranks.iloc[position] == args.top_rank]
    if args.problem_id:
        selected_ids = {str(value).strip().upper() for value in args.problem_id}
        positions = [
            position
            for position in positions
            if str(frame.iloc[position][id_column]).strip().upper() in selected_ids
        ]
    if not positions:
        raise FitDataError("没有符合筛选条件的公式行")

    data_subdirs = tuple(str(value).strip() for value in args.data_subdirs)
    if any(not value for value in data_subdirs):
        raise FitDataError("data-subdirs 中不能包含空名称")
    if len(data_subdirs) != len(set(data_subdirs)):
        raise FitDataError(f"data-subdirs 包含重复名称：{data_subdirs}")
    dataset_indices = tuple(args.dataset_indices)
    if any(index < 0 for index in dataset_indices):
        raise FitDataError("dataset-indices 不能包含负数")
    if len(dataset_indices) != len(set(dataset_indices)):
        raise FitDataError(f"dataset-indices 包含重复编号：{dataset_indices}")

    records = [(position, frame.iloc[position].to_dict()) for position in positions]
    tasks = [
        (position, row, subdir, subdir_position)
        for subdir_position, subdir in enumerate(data_subdirs)
        for position, row in records
    ]
    actual_workers = min(args.workers, len(tasks))
    print(
        f"开始拟合：{len(data_subdirs)} 个数据子目录 × {len(records)} 个公式 × "
        f"{len(dataset_indices)} 组数据 = "
        f"{len(tasks) * len(dataset_indices)} 个 CSV；并行进程={actual_workers}",
        flush=True,
    )
    results: list[FitResult] = []
    if actual_workers == 1:
        for done, (position, row, subdir, subdir_position) in enumerate(tasks, start=1):
            result = fit_row_subdir(
                position,
                row,
                str(data_root),
                args.data_template,
                subdir,
                subdir_position,
                dataset_indices,
                id_column,
                function_column,
                args.target_column,
                args.max_nfev,
                args.restarts,
                args.seed,
                args.failure_value,
            )
            results.append(result)
            print(
                f"[{done}/{len(tasks)}] {result.data_subdir}/{result.problem_id}: "
                f"{result.status}, "
                f"NMSE={result.nmse:.6g}, R2={result.r2:.6g}",
                flush=True,
            )
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=actual_workers, mp_context=context) as pool:
            futures = {
                pool.submit(
                    fit_row_subdir,
                    position,
                    row,
                    str(data_root),
                    args.data_template,
                    subdir,
                    subdir_position,
                    dataset_indices,
                    id_column,
                    function_column,
                    args.target_column,
                    args.max_nfev,
                    args.restarts,
                    args.seed,
                    args.failure_value,
                ): (position, subdir)
                for position, row, subdir, subdir_position in tasks
            }
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{done}/{len(tasks)}] {result.data_subdir}/{result.problem_id}: "
                    f"{result.status}, "
                    f"NMSE={result.nmse:.6g}, R2={result.r2:.6g}",
                    flush=True,
                )

    result_by_key = {
        (result.row_position, result.data_subdir): result for result in results
    }
    output_rows: list[dict[str, Any]] = []
    for position in positions:
        source_row = frame.iloc[position]
        row_results = [result_by_key[(position, subdir)] for subdir in data_subdirs]
        metadata_result = next(
            (result for result in row_results if result.variable_names),
            row_results[0],
        )
        output_row: dict[str, Any] = {
            "problem_id": metadata_result.problem_id,
            "top_rank": metadata_result.top_rank,
            "function": source_row.get(function_column),
            "formula": source_row.get("formula", source_row.get("simplified_formula")),
            "independent_variables": ",".join(metadata_result.variable_names),
            "target": metadata_result.target_name,
            "parameter_count": metadata_result.parameter_count,
        }
        for result in row_results:
            prefix = f"{args.metric_prefix}_{result.data_subdir}"
            per_dataset = [
                {
                    "dataset_index": dataset_index,
                    "MSE": metric[0],
                    "NMSE": metric[1],
                    "R2": metric[2],
                }
                for dataset_index, metric in zip(dataset_indices, result.dataset_metrics)
            ]
            output_row.update(
                {
                    f"{prefix}_MSE": result.mse,
                    f"{prefix}_NMSE": result.nmse,
                    f"{prefix}_R2": result.r2,
                    f"{prefix}_fitted_parameters": json.dumps(
                        result.fitted_parameters, ensure_ascii=False
                    ),
                    f"{prefix}_dataset_metrics": json.dumps(
                        per_dataset, ensure_ascii=False
                    ),
                    f"{prefix}_sample_counts": json.dumps(
                        result.sample_counts, ensure_ascii=False
                    ),
                    f"{prefix}_data_files": json.dumps(
                        result.data_files, ensure_ascii=False
                    ),
                    f"{prefix}_status": result.status,
                    f"{prefix}_error": result.error,
                }
            )
        output_rows.append(output_row)

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else default_output_path(table_path, args.metric_prefix)
    )
    if output_path == table_path:
        raise FitDataError("输出路径不能覆盖输入表格")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        write_output(pd.DataFrame(output_rows), output_path)

    failed = sum(result.status != "OK" for result in results)
    print(
        f"完成：数据子目录/问题任务成功 {len(results) - failed}，失败 {failed}；"
        f"输出：{output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FitDataError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2)
