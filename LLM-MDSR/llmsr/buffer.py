# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A multi-island experience buffer that implements the evolutionary algorithm."""
from __future__ import annotations

import profile
from collections.abc import Mapping, Sequence
import copy
import dataclasses
import time
import ast
import re
import textwrap

import sympy as sp

from typing import Any, Tuple, Mapping

from absl import logging
import numpy as np
import scipy

from llmsr import code_manipulation
from llmsr import config as config_lib

MAX_FORMULA_COMPLEXITY = 25
COMPLEXITY_WEIGHT = 0.04
LOSS_EPSILON = 1e-12
GLOBAL_GUIDANCE_PROBABILITY = 0.20

# 每次最多展示几个全局公式
GLOBAL_GUIDANCE_COUNT = 2

# 无效公式只展示最近几个
INVALID_PROMPT_COUNT = 3

Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]

# 等价性去重只是辅助步骤。超过该规模时跳过深度代数比较，避免
# sp.simplify(a - b)在三角化简或多项式GCD中长时间阻塞训练。
MAX_EQUIVALENCE_CHECK_OPS = 60


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        non_finites = set(logits[~np.isfinite(logits)])
        raise ValueError(f'`logits` contains non-finite value(s): {non_finites}')
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)

    result = scipy.special.softmax(logits / temperature, axis=-1)
    index = np.argmax(result)
    result[index] = 1 - np.sum(result[0:index]) - np.sum(result[index + 1:])
    return result

def _get_nmse_statistics(
        scores_per_test: ScoresPerTest
) -> dict[str, float]:

    scores = np.asarray(
        [
            scores_per_test[k]
            for k in sorted(scores_per_test.keys())
        ],
        dtype=float,
    )

    if len(scores) == 0:
        raise ValueError("scores_per_test is empty")

    # evaluate() returns -NMSE
    nmse = -scores

    worst_n = min(2, len(nmse))

    return {
        "mean_nmse": float(np.mean(nmse)),
        "min_nmse": float(np.min(nmse)),
        "max_nmse": float(np.max(nmse)),
        "median_nmse": float(np.median(nmse)),
        "std_nmse": float(np.std(nmse)),
        "worst2_mean_nmse": float(
            np.mean(np.sort(nmse)[-worst_n:])
        ),
    }

# def _reduce_score(scores_per_test: ScoresPerTest) -> float:
#     test_scores = [scores_per_test[k] for k in scores_per_test.keys()]
#     return sum(test_scores) / len(test_scores)

def get_score(
        stats: dict[str, float],
        formula_complexity: int | None,
) -> float:
    """同时考虑Worst-2 NMSE和公式复杂度。"""
    loss = max(
        float(stats["worst2_mean_nmse"]),
        LOSS_EPSILON,
    )

    if formula_complexity is None:
        return -float("inf")

    return (
        -np.log(loss)
        - COMPLEXITY_WEIGHT * formula_complexity
    )


# def get_score(stats: dict[str, float]) -> float:
#     return -stats["worst2_mean_nmse"]

_SYMPY_LOCALS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "log1p": lambda value: sp.log(1 + value),
    "expm1": lambda value: sp.exp(value) - 1,
    "abs": sp.Abs, "fabs": sp.Abs, "Abs": sp.Abs,
    "sign": sp.sign, "pi": sp.pi,
    # 常见NumPy函数。_ast_to_sympy会先移除np./numpy.前缀。
    "square": lambda value: value ** 2,
    "power": lambda base, exponent: base ** exponent,
    "minimum": sp.Min, "maximum": sp.Max,
    "where": lambda condition, true_value, false_value: sp.Piecewise(
        (true_value, condition),
        (false_value, True),
    ),
    "True": True, "False": False,
}


def _ast_to_sympy(node: ast.AST, variables: dict[str, Any]) -> Any:
    """将一个Python AST表达式转换为SymPy表达式。"""
    expression = ast.unparse(node)

    # np.cos、numpy.cos、math.cos → cos
    expression = re.sub(r"\b(?:np|numpy|math)\.", "", expression)

    # params[0]、x[1] → params_0、x_1
    expression = re.sub(
        r"\b([A-Za-z_]\w*)\s*\[\s*(\d+)\s*\]",
        r"\1_\2",
        expression,
    )

    # 未显式注册的新函数先保留为SymPy未定义函数，避免把函数名误建成
    # Symbol后在调用时出现“'Symbol' object is not callable”。
    called_function_names: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if isinstance(child.func, ast.Name):
            called_function_names.add(child.func.id)
        elif isinstance(child.func, ast.Attribute):
            called_function_names.add(child.func.attr)

    # 明确创建其余符号，避免E、I等名称被SymPy解释成内置常量。
    for name in re.findall(r"\b[A-Za-z_]\w*\b", expression):
        if name not in variables:
            if name in called_function_names:
                variables[name] = sp.Function(name)
            else:
                variables[name] = sp.Symbol(name)

    return sp.sympify(expression, locals=variables)


def _bind_assignment(
        target: ast.AST,
        value: Any,
        variables: dict[str, Any],
) -> bool:
    """把普通或解包赋值写入SymPy局部变量表。"""
    if isinstance(target, ast.Name):
        variables[target.id] = value
        return True

    if isinstance(target, (ast.Tuple, ast.List)):
        if not isinstance(value, (tuple, list, sp.Tuple)):
            return False
        if len(target.elts) != len(value):
            return False
        return all(
            _bind_assignment(child_target, child_value, variables)
            for child_target, child_value in zip(target.elts, value)
        )

    return False


def _program_to_sympy(program: code_manipulation.Function) -> sp.Expr | None:
    """处理return前的赋值语句，并返回最终的SymPy表达式。"""
    try:
        body = textwrap.dedent(program.body).strip()
        statements = ast.parse(body).body
        variables: dict[str, Any] = dict(_SYMPY_LOCALS)

        for statement in statements:
            # a = expression 或 a = b = expression
            if isinstance(statement, ast.Assign):
                value = _ast_to_sympy(statement.value, variables)

                for target in statement.targets:
                    if not _bind_assignment(target, value, variables):
                        return None

            # a: float = expression
            elif isinstance(statement, ast.AnnAssign):
                if not isinstance(statement.target, ast.Name) or statement.value is None:
                    return None
                variables[statement.target.id] = _ast_to_sympy(statement.value, variables)

            # a += expression、a *= expression等
            elif isinstance(statement, ast.AugAssign):
                if not isinstance(statement.target, ast.Name):
                    return None

                name = statement.target.id
                if name not in variables:
                    return None

                combined = ast.BinOp(
                    left=ast.Name(id=name, ctx=ast.Load()),
                    op=statement.op,
                    right=statement.value,
                )
                variables[name] = _ast_to_sympy(combined, variables)

            # 忽略单独出现的字符串，例如注释性docstring
            elif isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
                if not isinstance(statement.value.value, str):
                    return None

            # 返回已经代入前面变量的最终表达式
            elif isinstance(statement, ast.Return):
                if statement.value is None:
                    return None
                return _ast_to_sympy(statement.value, variables)

            # if、for、while等控制结构暂时无法直接转换为单一公式
            else:
                return None

    except Exception as error:
        logging.warning("Failed to convert program to SymPy: %s; body=%s",
                        error, program.body)
        return None

    return None


def _simplify_formula_and_get_complexity(
        program: code_manipulation.Function,
) -> tuple[sp.Expr | None, int | None]:
    """用SymPy化简公式，并以count_ops作为公式复杂度。"""
    expression = _program_to_sympy(program)
    if expression is None:
        logging.warning(
            "Formula complexity is unavailable because conversion failed; body=%s",
            program.body,
        )
        return None, None

    try:
        simplified_expression = sp.simplify(expression)
        complexity = int(sp.count_ops(simplified_expression, visual=False))
        return simplified_expression, complexity
    except Exception as error:
        logging.warning(
            "Failed to simplify formula or calculate complexity: %s; body=%s",
            error,
            program.body,
        )
        return None, None


def _same_formula(a: sp.Expr | None, b: sp.Expr | None) -> bool:
    if a is None or b is None:
        return False

    try:
        # 已经规范化为完全相同的表达式时无需再化简。
        if a == b:
            return True

        difference = a - b
        if difference == 0:
            return True

        # 对复杂公式调用simplify可能触发成本极高的trigsimp/factor/GCD。
        # 这种情况下宁可漏掉一次去重，也不能阻塞整个搜索过程。
        operation_count = int(sp.count_ops(difference, visual=False))
        if operation_count > MAX_EQUIVALENCE_CHECK_OPS:
            return False

        # expand_mul处理加减乘的简单重排；cancel处理低复杂度有理式。
        # 两者不会主动进入完整的深层三角恒等式搜索。
        difference = sp.expand_mul(difference)
        if difference == 0:
            return True

        difference = sp.cancel(difference)
        return difference == 0
    except Exception as error:
        logging.warning("Formula equivalence check failed: %s", error)
        return False
    
# def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
#     """Represents test scores as a canonical signature."""
#     return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))

def _get_signature(
        stats: dict[str, float],
        formula_complexity: int | None,
) -> Signature:
    # -1专门表示无法转换；正常公式的count_ops始终不小于0。
    complexity_signature = -1 if formula_complexity is None else formula_complexity
    return (
        round(stats["mean_nmse"], 3),
        round(stats["min_nmse"], 3),
        round(stats["max_nmse"], 3),
        complexity_signature,
    )

@dataclasses.dataclass(frozen=True)
class Prompt:
    """ A prompt produced by the Experience Buffer, to be sent to Samplers.

    Args:
      code: The prompt, ending with the header of the function to be completed.
      version_generated: The function to be completed is `_v{version_generated}`.
      island_id: Identifier of the island that produced the samples
                included in the prompt. Used to direct the newly generated sample
                into the same island.
    """
    code: str
    version_generated: int
    island_id: int


class ExperienceBuffer:
    """A collection of programs, organized as islands."""

    def __init__(
            self,
            config: config_lib.ExperienceBufferConfig,
            template: code_manipulation.Program,
            function_to_evolve: str,
    ) -> None:
        self._config: config_lib.ExperienceBufferConfig = config
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve

        # Initialize empty islands.
        self._islands: list[Island] = []
        for _ in range(config.num_islands):
            self._islands.append(
                Island(template, function_to_evolve, config.functions_per_prompt,
                       config.cluster_sampling_temperature_init,
                       config.cluster_sampling_temperature_period))
        self._best_score_per_island: list[float] = (
                [-float('inf')] * config.num_islands)
        self._best_program_per_island: list[code_manipulation.Function | None] = (
                [None] * config.num_islands)
        self._best_scores_per_test_per_island: list[ScoresPerTest | None] = (
                [None] * config.num_islands)
        
        self._top_k: int = 10
        self._top_programs: dict[str, dict[str, Any]] = {}
        self._invalid_programs: list[dict[str, Any]] = []
        self._last_reset_time: float = time.time()

    def _restore_empty_islands(self) -> None:
        """使用不同的全局优秀公式恢复空 island。"""

        top_programs = getattr(
            self,
            "_top_programs",
            {},
        )

        if not top_programs:
            return

        candidates = sorted(
            top_programs.values(),
            key=lambda record: record["score"],
            reverse=True,
        )[:self._top_k]

        empty_island_ids = [
            island_id
            for island_id, island in enumerate(self._islands)
            if island.is_empty
        ]

        if not empty_island_ids:
            return

        # 随机改变每次分配的起点。
        start_index = int(
            np.random.randint(len(candidates))
        )

        for offset, island_id in enumerate(empty_island_ids):
            record = candidates[
                (start_index + offset) % len(candidates)
            ]

            self._register_program_in_island(
                record["program"],
                island_id,
                record["scores_per_test"],
                record["stats"],
                record["score"],
                record.get("complexity"),
            )

    def _get_invalid_prompt(self) -> str:
        """要求大模型总结无效公式的共同原因并避免重复犯错。"""
        invalid_programs = getattr(self, "_invalid_programs", [])

        if not invalid_programs:
            return ""

        lines = [
            "# IMPORTANT FAILURE ANALYSIS:",
            "# The formulas listed below failed during execution or evaluation.",
            "# Before generating a new formula, analyze these failed formulas internally",
            "# and identify their common root causes.",
            "#",
            "# Pay particular attention to:",
            "# 1. Undefined variables or constants.",
            "# 2. Incorrect or out-of-range params indices.",
            "# 3. Intermediate variables used before assignment.",
            "# 4. Invalid NumPy operations, domains, shapes, or return types.",
            "# 5. Missing, non-numeric, NaN, or infinite return values.",
            "# 6. Calls to previous ancestor functions.",
            "#",
            "# The new formula must avoid all identified failure patterns.",
            "# Every symbol must be defined by the function arguments, params,",
            "# fixed constants, or earlier assignment statements.",
            "# Do not copy the failed formulas or reproduce their underlying mistakes.",
            "# Perform the failure analysis internally and output only the new valid code.",
            "#",
            "# FAILED FORMULAS:",
        ]

        for index, record in enumerate(invalid_programs[-10:], start=1):
            body = record["program"].body.strip()

            lines.append(f"# Failed formula {index}:")
            lines.extend(f"# {line}" for line in body.splitlines())
            lines.append("#")

        return "\n".join(lines)

    def _get_top_programs_prompt(self) -> str:
        """将全局最佳公式加入prompt，并要求模型进行实质性改进。"""
        top_programs = getattr(self, "_top_programs", {})

        if not top_programs:
            return ""

        pool = sorted(
            top_programs.values(),
            key=lambda record: record["score"],
            reverse=True,
        )[:self._top_k]

        sample_count = min(
            GLOBAL_GUIDANCE_COUNT,
            len(pool),
        )

        selected_indices = np.random.choice(
            len(pool),
            size=sample_count,
            replace=False,
        )

        records = [
            pool[int(index)]
            for index in selected_indices
        ]

        # 只控制显示顺序，不改变抽样结果
        records.sort(
            key=lambda record: record["score"],
            reverse=True,
        )

        lines = [
            "# GLOBAL REFERENCE FORMULAS:",
            "# The formulas below are randomly selected from the current",
            "# globally high-scoring formulas across all islands.",
            "# They are references for exploration, not templates that must be copied.",
            "# Analyze them internally to identify useful directions, including:",
            "# - effective mathematical structures and operator combinations;",
            "# - useful parameterizations and variable interactions;",
            "# - structures shared by several high-scoring formulas;",
            "# - weaknesses that may still be improved.",
            "#",
            "# Generate a genuinely new and improved formula following these directions.",
            "# Do not copy any original candidate or best formula.",
            "# Do not produce an algebraically equivalent version of any listed formula.",
            "# Renaming variables, reordering terms, factoring, expanding, or replacing",
            "# expressions with equivalent identities does not count as improvement.",
            "# The new formula must contain a meaningful structural or mathematical change.",
            "# Analyze internally and output only the new indented function body.",
            "#",
        ]

        for rank, record in enumerate(records, start=1):
            program = record["program"]
            stats = record["stats"]

            lines.append(
                f"# Reference formula {rank}: "
                f"score={record['score']:.8g}, "
                f"worst2_mean_nmse={stats['worst2_mean_nmse']:.8g}, "
                f"complexity={record.get('complexity')}"
            )
            lines.extend(f"# {line}" for line in program.body.strip().splitlines())
            lines.append("#")

        return "\n".join(lines)

    def get_prompt(self) -> Prompt:
        self._restore_empty_islands()

        available_islands = [
            island_id
            for island_id, island in enumerate(self._islands)
            if not island.is_empty
        ]

        if not available_islands:
            raise RuntimeError(
                "All islands are empty and no global best formula is available. "
                "The initial formula must evaluate successfully before sampling."
            )

        island_id = int(np.random.choice(available_islands))
        code, version_generated = self._islands[island_id].get_prompt()

        guidance = []

        # 大多数时候只依据当前 island 演化。
        if np.random.random() < GLOBAL_GUIDANCE_PROBABILITY:
            global_guidance = self._get_top_programs_prompt()

            if global_guidance:
                guidance.append(global_guidance)

        # invalid_guidance = self._get_invalid_prompt()

        # if invalid_guidance:
        #     guidance.append(invalid_guidance)

        guidance = "\n\n".join(guidance)

        if guidance:
            code = f"{guidance}\n\n{code}"

        return Prompt(code, version_generated, island_id)
    
    def register_invalid_program(self, program: code_manipulation.Function) -> None:
        """保存最近10个互不等价的无效公式。"""
        if not hasattr(self, "_invalid_programs"):
            self._invalid_programs = []

        new_expression = _program_to_sympy(program)

        for record in self._invalid_programs:
            old_program = record["program"]

            equivalent = program.body.strip() == old_program.body.strip()

            if not equivalent:
                old_expression = _program_to_sympy(old_program)
                equivalent = _same_formula(new_expression, old_expression)

            # 已有相同或等价公式，不添加，也不更新时间
            if equivalent:
                return

        self._invalid_programs.append({
            "program": copy.deepcopy(program),
            "score": None,
        })

        # 按加入顺序保留最近10个
        del self._invalid_programs[:-10]  


            
    def get_invalid_programs(self) -> list[dict[str, Any]]:
        """返回最近10个互不等价的无效公式，顺序为旧到新。"""
        if not hasattr(self, "_invalid_programs"):
            self._invalid_programs = []

        return copy.deepcopy(self._invalid_programs)
          
    def _ensure_top_program_storage(self) -> None:
        """兼容创建Top-10功能之前保存的旧checkpoint."""

        if not hasattr(self, "_top_k"):
            self._top_k = 10

        if not hasattr(self, "_top_programs"):
            self._top_programs = {}

        # 修复旧checkpoint中缺少或曾因函数映射不完整而为空的复杂度。
        for record in self._top_programs.values():
            if record.get("complexity") is not None:
                continue
            program = record.get("program")
            if program is None:
                continue
            _, complexity = _simplify_formula_and_get_complexity(program)
            record["complexity"] = complexity
            program.complexity = complexity

    def _print_current_global_best(self) -> None:
        """打印当前全局最高分公式的简化结果。"""
        if not self._top_programs:
            return

        best = max(
            self._top_programs.values(),
            key=lambda record: record["score"],
        )

        stats = best["stats"]

        print("\n" + "=" * 60)
        print("Current Global Best Simplified Formula")
        print("=" * 60)
        print("Formula     :", best.get("simplified_formula"))
        print("Score       :", best["score"])
        print("Complexity  :", best.get("complexity"))
        print(
            "Worst-2 NMSE:",
            stats.get("worst2_mean_nmse"),
        )
        print("=" * 60 + "\n")

    def _update_global_top_programs(
        self,
        program: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
        stats: dict[str, float],
        score: float,
        formula_complexity: int | None,
        simplified_expression: sp.Expr | None,
    ) -> None:
        """更新所有island共享的全局Top-10."""

        self._ensure_top_program_storage()

        new_body = program.body.strip()
        new_expression = simplified_expression

        equivalent_key = None
        equivalent_record = None

        # 最多只需要与当前10个公式比较
        for old_key, old_record in self._top_programs.items():
            old_program = old_record["program"]
            old_body = old_program.body.strip()

            # 先进行字符串比较
            equivalent = new_body == old_body

            # 字符串不同时，再使用SymPy进行代数比较
            if not equivalent and new_expression is not None:
                old_expression = _program_to_sympy(old_program)

                if old_expression is not None:
                    equivalent = _same_formula(
                        new_expression,
                        old_expression,
                    )

            if equivalent:
                equivalent_key = old_key
                equivalent_record = old_record
                break

        if equivalent_record is not None:
            # 已有等价公式的score更高或相同，不加入新公式
            if equivalent_record["score"] >= score:
                return

            # 新公式虽然等价，但score更高：
            # 删除旧记录，用新记录替换。
            # Top-10中的公式数量不会增加。
            del self._top_programs[equivalent_key]

        self._top_programs[new_body] = {
            "program": copy.deepcopy(program),
            "scores_per_test": dict(scores_per_test),
            "stats": dict(stats),
            "score": float(score),
            "complexity": formula_complexity,
            "simplified_formula": (
                str(simplified_expression)
                if simplified_expression is not None
                else None
            ),            
        }

        # score从高到低排序
        sorted_items = sorted(
            self._top_programs.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )


        # 不足10个时保留全部，超过10个时保留前10个
        self._top_programs = dict(
            sorted_items[:self._top_k]
        )
        self._print_current_global_best()        
    def get_top_programs(self) -> list[dict[str, Any]]:
        """返回按照score从高到低排列的全局Top-10."""

        self._ensure_top_program_storage()

        records = sorted(
            self._top_programs.values(),
            key=lambda record: record["score"],
            reverse=True,
        )

        return copy.deepcopy(records)
    
    def _register_program_in_island(
            self,
            program: code_manipulation.Function,
            island_id: int,
            scores_per_test: ScoresPerTest,
            stats: dict[str, float],
            score: float,
            formula_complexity: int | None,
            **kwargs,
    ) -> None:
        """Registers `program` in the specified island."""

        self._islands[island_id].register_program(
            program,
            scores_per_test,
            stats,
            formula_complexity,
            score = score
        )

        if score > self._best_score_per_island[island_id]:
            self._best_program_per_island[island_id] = program
            self._best_scores_per_test_per_island[island_id] = scores_per_test
            self._best_score_per_island[island_id] = score
            logging.info('Best score of island %d increased to %s', island_id, score)

        profiler: profile.Profiler = kwargs.get('profiler', None)
        if profiler:
            global_sample_nums = kwargs.get('global_sample_nums', None)
            sample_time = kwargs.get('sample_time', None)
            evaluate_time = kwargs.get('evaluate_time', None)
            program.score = score
            program.global_sample_nums = global_sample_nums
            program.sample_time = sample_time
            program.evaluate_time = evaluate_time
            program.mean_nmse = stats["mean_nmse"]
            program.min_nmse = stats["min_nmse"]
            program.max_nmse = stats["max_nmse"]
            program.median_nmse = stats["median_nmse"]
            program.std_nmse = stats["std_nmse"]
            program.worst2_mean_nmse = stats["worst2_mean_nmse"]
            # Profiler写JSON时可直接序列化该属性为"complexity"字段。
            program.complexity = formula_complexity
            program.nmse_per_test = {
                k: -float(scores_per_test[k])
                for k in sorted(scores_per_test.keys())
            }
            print("\n[DEBUG BEFORE PROFILER]")
            print("global_sample_nums =", program.global_sample_nums)
            print("sample_time =", program.sample_time)
            print("evaluate_time =", program.evaluate_time)
            print("score =", program.score)
            print("mean_nmse =", program.mean_nmse)
            # print("min_nmse =", program.min_nmse)
            # print("max_nmse =", program.max_nmse)
            # print("median_nmse =", program.median_nmse)
            # print("std_nmse =", program.std_nmse)
            # print("worst2_mean_nmse =", program.worst2_mean_nmse)
            print("complexity =", program.complexity)
            # print("nmse_per_test =", program.nmse_per_test)
            print("[END DEBUG]\n")

            profiler.register_function(program)


    def register_program(
            self,
            program: code_manipulation.Function,
            island_id: int | None,
            scores_per_test: ScoresPerTest,
            **kwargs,
    ) -> None:
        """注册公式，同时更新所有island共享的全局Top-10."""

        # 每个新公式只计算一次统计量
        stats = _get_nmse_statistics(scores_per_test)
        simplified_expression, formula_complexity = (
            _simplify_formula_and_get_complexity(program)
        )
        if (formula_complexity is None) :
            logging.info(
                "Rejecting formula: complexity=%s, limit=%d",
                formula_complexity,
                MAX_FORMULA_COMPLEXITY,
            )
            return
        score = get_score(stats, formula_complexity)

        # 即使未启用profiler，也让checkpoint和其他JSON序列化逻辑能读取复杂度。
        program.complexity = formula_complexity

        # 不论公式来自哪个island，都进入同一个全局Top-10
        # 当island_id=None时，这里也只执行一次
        self._update_global_top_programs(
            program=program,
            scores_per_test=scores_per_test,
            stats=stats,
            score=score,
            formula_complexity=formula_complexity,
            simplified_expression=simplified_expression,
        )

        if island_id is None:
            for target_island_id in range(len(self._islands)):
                self._register_program_in_island(
                    program=program,
                    island_id=target_island_id,
                    scores_per_test=scores_per_test,
                    stats=stats,
                    score=score,
                    formula_complexity=formula_complexity,
                    **kwargs,
                )
        else:
            self._register_program_in_island(
                program=program,
                island_id=island_id,
                scores_per_test=scores_per_test,
                stats=stats,
                score=score,
                formula_complexity=formula_complexity,
                **kwargs,
            )

        # Check island reset
        if time.time() - self._last_reset_time > self._config.reset_period:
            self._last_reset_time = time.time()
            self.reset_islands()


    def reset_islands(self) -> None:
        """Resets the weaker half of islands."""
        # Sort best scores after adding minor noise to break ties.
        indices_sorted_by_score: np.ndarray = np.argsort(
            self._best_score_per_island +
            np.random.randn(len(self._best_score_per_island)) * 1e-6)
        num_islands_to_reset = self._config.num_islands // 2
        reset_islands_ids = indices_sorted_by_score[:num_islands_to_reset]
        keep_islands_ids = indices_sorted_by_score[num_islands_to_reset:]
        for island_id in reset_islands_ids:
            self._islands[island_id] = Island(
                self._template,
                self._function_to_evolve,
                self._config.functions_per_prompt,
                self._config.cluster_sampling_temperature_init,
                self._config.cluster_sampling_temperature_period)
            self._best_score_per_island[island_id] = -float('inf')
            founder_island_id = np.random.choice(keep_islands_ids)
            founder = self._best_program_per_island[founder_island_id]
            founder_scores = self._best_scores_per_test_per_island[founder_island_id]

            founder_stats = _get_nmse_statistics(founder_scores)
            _, founder_complexity = _simplify_formula_and_get_complexity(founder)
            founder_score = get_score(founder_stats, founder_complexity)


            self._register_program_in_island(
                program=founder,
                island_id=island_id,
                scores_per_test=founder_scores,
                stats=founder_stats,
                score=founder_score,
                formula_complexity=founder_complexity,
            )


class Island:
    """A sub-population of the program skeleton experience buffer."""

    def __init__(
            self,
            template: code_manipulation.Program,
            function_to_evolve: str,
            functions_per_prompt: int,
            cluster_sampling_temperature_init: float,
            cluster_sampling_temperature_period: int,
    ) -> None:
        self._template: code_manipulation.Program = template
        self._function_to_evolve: str = function_to_evolve
        self._functions_per_prompt: int = functions_per_prompt
        self._cluster_sampling_temperature_init = cluster_sampling_temperature_init
        self._cluster_sampling_temperature_period = (
            cluster_sampling_temperature_period)

        self._clusters: dict[Signature, Cluster] = {}
        self._num_programs: int = 0
        self._complexity_signatures_ready = True

    @property
    def is_empty(self) -> bool:
        return not self._clusters

    def _ensure_complexity_signatures(self) -> None:
        """把旧checkpoint中的三维island签名迁移为含复杂度的四维签名。"""
        if getattr(self, "_complexity_signatures_ready", False):
            return

        migrated_clusters: dict[Signature, Cluster] = {}
        for old_signature, old_cluster in self._clusters.items():
            score_signature = tuple(old_signature[:3])
            for program in old_cluster._programs:
                complexity = getattr(program, "complexity", None)
                if complexity is None:
                    _, complexity = _simplify_formula_and_get_complexity(program)
                    program.complexity = complexity

                complexity_signature = -1 if complexity is None else complexity
                new_signature = score_signature + (complexity_signature,)
                if new_signature not in migrated_clusters:
                    migrated_clusters[new_signature] = Cluster(
                        old_cluster.score,
                        program,
                    )
                else:
                    migrated_clusters[new_signature].register_program(program)

        self._clusters = migrated_clusters
        self._complexity_signatures_ready = True

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
            stats: dict[str, float],
            formula_complexity: int | None,
            score: float,
    ) -> None:

        self._ensure_complexity_signatures()

        signature = _get_signature(
            stats,
            formula_complexity,
        )

        if signature not in self._clusters:
            self._clusters[signature] = Cluster(
                score,
                program,
            )
        else:
            self._clusters[signature].register_program(program)

        self._num_programs += 1


    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""
        self._ensure_complexity_signatures()
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])
        
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures),
            size=functions_per_prompt,
            replace=False,
            p=probabilities,
        )
        chosen_signatures = [signatures[i] for i in idx]
        implementations = []
        scores = []
        for signature in chosen_signatures:
            cluster = self._clusters[signature]
            implementations.append(cluster.sample_program())
            scores.append(cluster.score)

        indices = np.argsort(scores)
        sorted_implementations = [implementations[i] for i in indices]
        version_generated = len(sorted_implementations) + 1
        return self._generate_prompt(sorted_implementations), version_generated


    def _generate_prompt(
        self,
        implementations: Sequence[code_manipulation.Function]) -> str:
        """Create a prompt containing a sequence of function implementations."""

        implementations = copy.deepcopy(implementations)

        # Format the names and docstrings of functions to be included in the prompt.
        versioned_functions: list[code_manipulation.Function] = []

        for i, implementation in enumerate(implementations):

            new_function_name = f'{self._function_to_evolve}_v{i}'

            implementation.name = new_function_name

            # Update the docstring for all subsequent functions after `_v0`.
            if i >= 1:
                implementation.docstring = (
                    f'Improved version of '
                    f'`{self._function_to_evolve}_v{i - 1}`.'
                )

            # If the function is recursive, replace calls to itself
            # with its new versioned name.
            implementation = code_manipulation.rename_function_calls(
                str(implementation),
                self._function_to_evolve,
                new_function_name,
            )

            versioned_functions.append(
                code_manipulation.text_to_function(implementation)
            )

        # ------------------------------------------------------------
        # Create the next function that the LLM must improve
        # ------------------------------------------------------------

        next_version = len(implementations)

        new_function_name = (
            f'{self._function_to_evolve}_v{next_version}'
        )

        header = dataclasses.replace(
            implementations[-1],

            name=new_function_name,

            # Give the LLM the previous valid body instead of
            # an empty function.
            body=implementations[-1].body,

            docstring=(
                f'Improved version of '
                f'`{self._function_to_evolve}_v{next_version - 1}`. '
                'Rewrite the mathematical expression in the function body. '
                'Return a complete valid implementation with an explicit '
                'return statement.'
            ),
        )

        versioned_functions.append(header)

        # Replace functions in the template with the constructed versions.
        prompt = dataclasses.replace(
            self._template,
            functions=versioned_functions,
        )

        return str(prompt)


class Cluster:
    """ A cluster of programs on the same island and with the same Signature. """

    def __init__(self, score: float, implementation: code_manipulation.Function):
        self._score = score
        self._programs: list[code_manipulation.Function] = [implementation]
        self._lengths: list[int] = [len(str(implementation))]

    @property
    def score(self) -> float:
        return self._score

    def register_program(self, program: code_manipulation.Function) -> None:
        """Adds `program` to the cluster."""
        self._programs.append(program)
        self._lengths.append(len(str(program)))

    def sample_program(self) -> code_manipulation.Function:
        """Samples a program, giving higher probability to shorther programs."""
        normalized_lengths = (np.array(self._lengths) - min(self._lengths)) / (
                max(self._lengths) + 1e-6)
        probabilities = _softmax(-normalized_lengths, temperature=1.0)
        return np.random.choice(self._programs, p=probabilities)
