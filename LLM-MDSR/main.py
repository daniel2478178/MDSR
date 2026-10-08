global_max_sample_num = 12 

import multiprocessing as mp
import re
import traceback
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from pathlib import Path
from paths import CSV_ROOT, PROJECT_ROOT


PROBLEM_IDS = [f"P{index:02d}" for index in range(1, 60)]


def normalize_problem_id(value: str) -> str:
    """Normalize p10/P10 to P10 and reject malformed IDs."""
    problem_id = value.strip().upper()
    if (
        re.fullmatch(r"P\d{2}", problem_id) is None
        or not 1 <= int(problem_id[1:]) <= 59
    ):
        raise ValueError(
            "problem_id must have the form P01, P02, ..., P59"
        )
    return problem_id


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=True)
parser.add_argument('--api_model', type=str, default="deepseek-v4-flash")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument(
    '--num_processes',
    type=int,
    default=1,
    help='Maximum number of problems to run concurrently (default: 4)',
)
args = parser.parse_args()

# 新增了一个函数来读取数据集
def read_dataset(data_dir, spec_path):
    dataset = {}
    for index in range(8):
        csv_file = data_dir / f"{index}.csv"

        if not csv_file.is_file():
            raise FileNotFoundError(f"Dataset not found: {csv_file}")

        df = pd.read_csv(csv_file)

        numeric = df.apply(pd.to_numeric, errors="coerce")
        values = numeric.to_numpy(dtype=float)

        valid = np.all(np.isfinite(values), axis=1)
        values = values[valid]

        if len(values) == 0:
            raise ValueError(f"No finite data in {csv_file}")

        X = values[:, :-1]
        y = values[:, -1]

        if "torch" in str(spec_path).lower():
            import torch
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        dataset[f"dataset_{index}"] = {
            "inputs": X,
            "outputs": y,
        }
    return dataset


def run(problem_id: str) -> str:
    """Run one LLMSR problem, for example run("p07")."""
    problem_id = normalize_problem_id(problem_id)

    # Derive every problem-specific value from problem_id.
    spec_path = PROJECT_ROOT / "prompts" / f"{problem_id}.txt"
    log_path = CSV_ROOT / "logs" / problem_id
    problem_name = problem_id
    data_dir = CSV_ROOT / "train_data" / problem_id

    from llmsr import pipeline
    from llmsr import config
    from llmsr import sampler
    from llmsr import evaluator

    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config = config.Config(use_api = args.use_api, 
                           api_model = args.api_model,)

    print(f"Problem ID : {problem_id}")
    print(f"Problem    : {problem_name}")
    print(f"Prompt     : {spec_path}")
    print(f"Log path   : {log_path}")
    print(f"Data dir   : {data_dir}")

    # Load prompt specification
    with spec_path.open(encoding="utf-8") as f:
        specification = f.read()
    
    # Load dataset
    dataset = read_dataset(
        data_dir=data_dir,
        spec_path=spec_path,
    )
    
    
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        # log_dir = 'logs/m1jobs-mixtral-v10',
        log_dir=str(log_path),
    )

    return problem_id


def run_safely(problem_id: str) -> tuple[str, str | None]:
    """Run one problem in one child process and return any traceback."""
    try:
        finished_problem = run(problem_id)
        return finished_problem, None
    except Exception:
        return problem_id, traceback.format_exc()


def run_all(problem_ids: list[str], num_processes: int) -> None:
    """Run problems in isolated child processes with bounded concurrency."""
    if num_processes < 1:
        raise ValueError("num_processes must be at least 1")

    normalized_ids = [normalize_problem_id(item) for item in problem_ids]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("problem_ids contains duplicate IDs")

    total = len(normalized_ids)
    completed = 0
    failures: list[tuple[str, str]] = []

    print(
        f"Starting {total} problems with at most "
        f"{num_processes} concurrent processes.",
        flush=True,
    )

    # ProcessPoolExecutor workers are non-daemonic, so LLMSR's LocalSandbox
    # may safely create its own child process inside a worker. Pending problem
    # IDs form a work queue; a free worker automatically takes the next ID.
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {
            executor.submit(run_safely, problem_id): problem_id
            for problem_id in normalized_ids
        }

        for future in as_completed(futures):
            submitted_problem_id = futures[future]

            try:
                problem_id, error = future.result()
            except Exception:
                problem_id = submitted_problem_id
                error = traceback.format_exc()

            completed += 1

            if error is None:
                status = "completed"
            else:
                status = "failed"
                failures.append((problem_id, error))

            print(
                f"[overall {completed}/{total}] {problem_id}: {status}",
                flush=True,
            )

    if failures:
        print("\nFailed problems:", flush=True)
        for problem_id, error in failures:
            print("=" * 80, flush=True)
            print(f"{problem_id}\n{error}", flush=True)

        raise RuntimeError(
            f"{len(failures)} of {total} problems failed"
        )

    print(f"All {total} problems completed successfully.", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    run_all(
        problem_ids=PROBLEM_IDS,
        num_processes=args.num_processes,
    )
