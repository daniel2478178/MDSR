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


Signature = Tuple[float, ...]
ScoresPerTest = Mapping[Any, float]


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
def _reduce_score(
        scores_per_test: ScoresPerTest
) -> float:

    stats = _get_nmse_statistics(
        scores_per_test
    )

    return -stats["mean_nmse"]
def get_score(stats: dict[str, float]) -> float:
    return -stats["worst2_mean_nmse"]

_SYMPY_LOCALS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt,
    "abs": sp.Abs, "Abs": sp.Abs, "pi": sp.pi,
}


def _ast_to_sympy(node: ast.AST, variables: dict[str, Any]) -> sp.Expr:
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

    # 明确创建其余符号，避免E、I等名称被SymPy解释成内置常量
    for name in re.findall(r"\b[A-Za-z_]\w*\b", expression):
        if name not in variables:
            variables[name] = sp.Symbol(name)

    return sp.sympify(expression, locals=variables)


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
                    if not isinstance(target, ast.Name):
                        return None
                    variables[target.id] = value

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


def _same_formula(a: sp.Expr | None, b: sp.Expr | None) -> bool:
    if a is None or b is None:
        return False

    try:
        difference = sp.simplify(a - b)
        return difference == 0 or difference.equals(0) is True
    except Exception:
        return False
    
# def _get_signature(scores_per_test: ScoresPerTest) -> Signature:
#     """Represents test scores as a canonical signature."""
#     return tuple(scores_per_test[k] for k in sorted(scores_per_test.keys()))

def _get_signature(stats: dict[str, float]) -> Signature:
    return (
        round(stats["mean_nmse"], 3),
        round(stats["min_nmse"], 3),
        round(stats["max_nmse"], 3),
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
        top_programs = getattr(self, "_top_programs", {})

        if not top_programs:
            return

        best = max(top_programs.values(), key=lambda record: record["score"])

        for island_id, island in enumerate(self._islands):
            if island.is_empty:
                self._register_program_in_island(
                    best["program"],
                    island_id,
                    best["scores_per_test"],
                    best["stats"],
                    best["score"],
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

        records = sorted(
            top_programs.values(),
            key=lambda record: record["score"],
            reverse=True,
        )[:10]

        lines = [
            "# GLOBAL BEST FORMULAS:",
            "# The formulas below currently have the highest evaluation scores",
            "# across all islands.",
            "#",
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
                f"# Best formula {rank}: "
                f"score={record['score']:.8g}, "
                f"worst2_mean_nmse={stats['worst2_mean_nmse']:.8g}"
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

        guidance = [
            self._get_top_programs_prompt(),
            self._get_invalid_prompt(),
        ]
        guidance = "\n\n".join(section for section in guidance if section)

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
        print(f"[DEBUG] Current invalid programs (count={len(self._invalid_programs)}):")
        for i, record in enumerate(self._invalid_programs):
            print(f"Rank {i + 1}: program body={record['program'].body.strip()}")

            
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
    def _update_global_top_programs(
        self,
        program: code_manipulation.Function,
        scores_per_test: ScoresPerTest,
        stats: dict[str, float],
        score: float,
    ) -> None:
        """更新所有island共享的全局Top-10."""

        self._ensure_top_program_storage()

        new_body = program.body.strip()
        new_expression = _program_to_sympy(program)

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
        }

        # score从高到低排序
        sorted_items = sorted(
            self._top_programs.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )
        print(f"[DEBUG] Current global top programs (count={len(sorted_items)}):")
        for i, (key, record) in enumerate(sorted_items):
            print(f"Rank {i + 1}: score={record['score']}, program body={key}")     

        # 不足10个时保留全部，超过10个时保留前10个
        self._top_programs = dict(
            sorted_items[:self._top_k]
        )        
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
            **kwargs,
    ) -> None:
        """Registers `program` in the specified island."""

        self._islands[island_id].register_program(
            program,
            scores_per_test,
            stats,
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
            program.nmse_per_test = {
                k: -float(scores_per_test[k])
                for k in sorted(scores_per_test.keys())
            }
            print("\n[DEBUG BEFORE PROFILER]")
            print("score =", program.score)
            print("mean_nmse =", program.mean_nmse)
            print("min_nmse =", program.min_nmse)
            print("max_nmse =", program.max_nmse)
            print("median_nmse =", program.median_nmse)
            print("std_nmse =", program.std_nmse)
            print("worst2_mean_nmse =", program.worst2_mean_nmse)
            print("nmse_per_test =", program.nmse_per_test)
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
        score = get_score(stats)

        # 不论公式来自哪个island，都进入同一个全局Top-10
        # 当island_id=None时，这里也只执行一次
        self._update_global_top_programs(
            program=program,
            scores_per_test=scores_per_test,
            stats=stats,
            score=score,
        )

        if island_id is None:
            for target_island_id in range(len(self._islands)):
                self._register_program_in_island(
                    program=program,
                    island_id=target_island_id,
                    scores_per_test=scores_per_test,
                    stats=stats,
                    score=score,
                    **kwargs,
                )
        else:
            self._register_program_in_island(
                program=program,
                island_id=island_id,
                scores_per_test=scores_per_test,
                stats=stats,
                score=score,
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
        founder_score = get_score(founder_stats)

        self._register_program_in_island(
            program=founder,
            island_id=island_id,
            scores_per_test=founder_scores,
            stats=founder_stats,
            score=founder_score,
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

    @property
    def is_empty(self) -> bool:
        return not self._clusters
    def _restore_empty_islands(self) -> None:
        top_programs = getattr(self, "_top_programs", {})

        if not top_programs:
            return

        best = max(top_programs.values(), key=lambda record: record["score"])

        for island_id, island in enumerate(self._islands):
            if island.is_empty:
                self._register_program_in_island(
                    best["program"],
                    island_id,
                    best["scores_per_test"],
                    best["stats"],
                    best["score"],
                )

    def register_program(
            self,
            program: code_manipulation.Function,
            scores_per_test: ScoresPerTest,
            stats: dict[str, float],
    ) -> None:

        signature = (
            round(stats["mean_nmse"], 3),
            round(stats["min_nmse"], 3),
            round(stats["max_nmse"], 3),
        )

        if signature not in self._clusters:
            score = get_score(stats)
            self._clusters[signature] = Cluster(score, program)
        else:
            self._clusters[signature].register_program(program)

        self._num_programs += 1


    def get_prompt(self) -> tuple[str, int]:
        """Constructs a prompt containing equation program skeletons from this island."""
        signatures = list(self._clusters.keys())
        cluster_scores = np.array(
            [self._clusters[signature].score for signature in signatures])
        
        period = self._cluster_sampling_temperature_period
        temperature = self._cluster_sampling_temperature_init * (
                1 - (self._num_programs % period) / period)
        probabilities = _softmax(cluster_scores, temperature)

        functions_per_prompt = min(len(self._clusters), self._functions_per_prompt)

        idx = np.random.choice(
            len(signatures), size=functions_per_prompt, p=probabilities)
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
