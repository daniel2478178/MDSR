
import os
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=True)
parser.add_argument('--api_model', type=str, default="deepseek-v4-flash")
parser.add_argument('--spec_path', type=str,default="d:/PhysicsMDRS_Dataset/prompts/p10.txt")
parser.add_argument('--log_path', type=str, default="d:/PhysicsMDRS_Dataset/physicsMDSR_Range_CSV/logs/p10")
parser.add_argument('--problem_name', type=str, default="p10")
parser.add_argument('--data_dir', type=str, default="d:/PhysicsMDRS_Dataset/physicsMDSR_Range_CSV/data/P10")
parser.add_argument('--run_id', type=int, default=1)
args = parser.parse_args()
# 新增了一个函数来读取数据集
def read_dataset(data_dir):
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

        if "torch" in args.spec_path.lower():
            import torch
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

        dataset[f"dataset_{index}"] = {
            "inputs": X,
            "outputs": y,
        }
    return dataset

if __name__ == '__main__':
    from llmsr import pipeline
    from llmsr import config
    from llmsr import sampler
    from llmsr import evaluator

    # Load config and parameters
    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config = config.Config(use_api = args.use_api, 
                           api_model = args.api_model,)
    global_max_sample_num = 10000 

    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    # Load dataset
    problem_name = args.problem_name
    dataset = read_dataset(data_dir=Path(args.data_dir))
    
    
    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        # log_dir = 'logs/m1jobs-mixtral-v10',
        log_dir=args.log_path,
    )
