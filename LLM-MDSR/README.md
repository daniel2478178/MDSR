# LLM-MDSR

LLM-MDSR uses a large language model to discover one symbolic equation structure
across several related datasets. The mathematical form stays the same, while
its numerical parameters are fitted separately for each dataset:

```text
dataset k: target = f(independent_variables, params_k)
```

This implementation is adapted from
**[LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR)**, the official
implementation of
[LLM-SR: Scientific Equation Discovery via Programming with Large Language Models](https://arxiv.org/abs/2404.18400).
It extends the LLM-SR search framework with multi-dataset evaluation, shared-formula
ranking, checkpoint recovery, and benchmark analysis tools. Please acknowledge
the original LLM-SR work when using this implementation.

The current runner targets physics problems `P01`–`P59`. Default paths resolve
from the repository location, so the runner and analysis tools can be launched
from any working directory.

## Workflow

1. Load a problem-specific prompt and eight datasets, `0.csv`–`7.csv`.
2. Ask the LLM for NumPy equation functions with dataset-specific `params[i]`.
3. Fit each candidate's parameters independently on each dataset using the
   evaluator in the prompt. The multi-dataset NumPy prompts use SciPy BFGS.
4. Rank candidates using fit error and symbolic complexity, remove equivalent
   formulas, and maintain an experience buffer with multiple search islands.
5. Save checkpoints, per-sample JSON records, and TensorBoard logs.
6. Export selected formulas and refit them on noise or distribution-shift data.

For prompts returning negative normalized mean squared error (NMSE), the current
ranking score is:

```text
score = -log(max(worst2_mean_nmse, 1e-12)) - 0.04 * complexity
```

Higher scores are better. `worst2_mean_nmse` is the average of the two largest
NMSE values returned by evaluation. Summaries use the datasets with valid
returned scores; inspect `nmse_per_test` to confirm coverage of all eight
datasets. The current symbolic complexity limit is 25.

## Environment

Run the following from the repository root to create a Python 3.11 environment
for API-based discovery and analysis:

```bash
cd LLM-MDSR
conda create -n llm-mdsr python=3.11 pip
conda activate llm-mdsr
python -m pip install numpy pandas scipy sympy matplotlib openpyxl requests absl-py torch tensorboard
```

The API workflow performs numerical fitting locally and does not require a
local LLM or CUDA GPU. PyTorch is still imported by the TensorBoard profiler.

`requirements.txt` pins CUDA 11.8 PyTorch packages, and `environment.yml` contains
Linux-specific package builds and NVIDIA dependencies. These files describe the
inherited GPU environment and are unsuitable for direct installation on Apple
Silicon. The API environment above avoids those platform-specific pins.

## Data and prompts

`paths.py` defines the shared defaults using `Path(__file__).resolve()`. The
training layout is:

```text
MDSR/
├── LLM-MDSR/
│   ├── main.py
│   └── paths.py
├── MDSR-PySR/
│   └── physicsMDSR_Range.xlsx
└── synthetic_data/
    ├── PhysicsMDRS_Synthetic_Dataset/
    │   ├── prompts/
    │   │   └── P01.txt ... P59.txt
    │   └── physicsMDSR_Range_CSV/
    │       ├── train_data/
    │       │   └── P01/ ... P59/     each containing 0.csv ... 7.csv
    │       └── logs/                created during search
    └── test_data/                   separately prepared robustness inputs
```

The repository includes the training CSVs and prompts under
`synthetic_data/PhysicsMDRS_Synthetic_Dataset/`, along with the example inputs
in `LLM-MDSR/data/`. A clone contains these datasets. Robustness inputs under
`synthetic_data/test_data/` must be prepared separately. Change the constants in
`paths.py` to use another layout. Explicit CLI path arguments in analysis tools
are resolved relative to your current working directory.

Every CSV must have a header, with independent variables first and the target
last. Input order must match the prompt's column assignments. The reader
converts values to numbers and drops rows containing non-finite values; all
eight files must exist and each must retain at least one valid row. Files
`8.csv`–`15.csv`, when present, are not consumed by this runner.

A prompt must contain exactly one function decorated with `@evaluate.run` and
one with `@equation.evolve`. The evaluation function fits the candidate's
parameters; the equation function supplies the initial structure to evolve.

To regenerate training CSVs, use the PySR benchmark workbook
after initializing that submodule. Run this from `LLM-MDSR/`:

```bash
python ../MDSR-PySR/scripts/data/generate_physics_mdsr_csv_openpyxl.py \
  ../MDSR-PySR/physicsMDSR_Range.xlsx \
  ../synthetic_data/PhysicsMDRS_Synthetic_Dataset/physicsMDSR_Range_CSV/train_data \
  --formula-column GenerationFormula --samples 5000 --groups 16
```

To generate prompts from the benchmark workbook:

```bash
python generate_prompts.py \
  ../MDSR-PySR/physicsMDSR_Range.xlsx \
  data/p20/prompt.txt \
  --sheet "Sampling design" \
  --output-dir ../synthetic_data/PhysicsMDRS_Synthetic_Dataset/prompts
```

The generator requires `ID`, `Target`, and comma-separated `IndependentVars`
columns. It updates variable names, signatures, column assignments, and the
initial equation, while preserving the template's other text. Review physical
hints for each problem. Existing prompt files in the output directory are
overwritten.

## Run with an API

The current sampler sends requests to `api.deepseek.com` at
`/v1/chat/completions` and reads its credential from `API_KEY`:

```bash
export API_KEY="YOUR_API_KEY"
python main.py --api_model YOUR_MODEL --num_processes 1
```

Choose a model available from the configured provider. `--api_model` changes
the request's model name; the provider endpoint and payload are configured in
`llmsr/sampler.py`.

The command processes all IDs in `PROBLEM_IDS`, currently `P01`–`P59`. To start
with one problem, set that list to the desired ID in `main.py`. There is no
problem-selection CLI option.

| Setting | Location | Current value or behavior |
| --- | --- | --- |
| Dataset and analysis paths | `paths.py` | Relative to the repository location |
| Problem IDs | `main.py`: `PROBLEM_IDS` | `P01`–`P59` |
| Sample budget per problem | `main.py`: `global_max_sample_num` | 12 generated samples |
| Concurrent problems | `--num_processes` | 1 by default |
| API model | `--api_model` | `deepseek-v4-flash` by default |
| API/local selection | `main.py`: `--use_api` default | API enabled |
| Samples per prompt | `llmsr/config.py` | 4 |
| Evaluation timeout | `llmsr/config.py` | 30 seconds per dataset evaluation |
| Search islands | `llmsr/config.py` | 20 |

Inspect the current CLI with `python main.py --help`. This runner does not
accept upstream LLM-SR's `--problem_name`, `--spec_path`, or `--log_path` options.
`run_llmsr.sh` retains commented upstream examples using those old options.
The parsed `--port` and `--run_id` values are currently unused.

## Local LLM serving

`llm_engine/engine.py` provides a Hugging Face model server. It additionally
requires `transformers`, `accelerate`, `flask`, and `flask-cors`; its optional
4-bit quantization path uses `bitsandbytes`.

On a suitable model-serving machine:

```bash
python llm_engine/engine.py \
  --model_path YOUR_MODEL_PATH \
  --gpu_ids 0 --host 127.0.0.1 --port 5000
```

The sampler's local endpoint defaults to
`http://127.0.0.1:5000/completions`. Configure its URL in `llmsr/sampler.py` if
the server runs elsewhere.

To select local sampling, change the `--use_api` default in `main.py` to
`False`, then run `python main.py --num_processes 1`. The current argument uses
`type=bool`, so passing `--use_api False` evaluates to true and does not disable
API sampling. `run_server.sh` contains an inherited Mixtral server example;
choose a model and serving configuration appropriate for your hardware.

## Outputs and resume behavior

Each problem writes to `CSV_ROOT/logs/<ID>/`, where `CSV_ROOT` is derived from
`PROJECT_ROOT`:

```text
logs/P01/
├── checkpoint.pkl             experience buffer, sample count, NumPy RNG state
├── samples/
│   └── samples_<order>.json    function, score, complexity, NMSE, timings
└── events.out.tfevents.*       TensorBoard events
```

Checkpoints are saved every 10 generated samples and when the sample budget is
reached. Rerunning the same problem resumes its checkpoint automatically. If
the saved count already reaches the configured budget, sampling stops; increase
the budget to continue or use a separate log directory for a new experiment.
`--run_id` does not create a separate run directory. Archived artifacts under
this repository's `logs/` and `backup/` are not automatically selected by
`main.py`.

View the active experiment logs with:

```bash
tensorboard --logdir ../synthetic_data/PhysicsMDRS_Synthetic_Dataset/physicsMDSR_Range_CSV/logs
```

## Export and evaluation

Export the highest-scoring retained formula for each problem:

```bash
python collect_top_programs_from_checkpoints.py \
  --top-n 1
```

Refit exported functions on noise or distribution-shift datasets:

```bash
python fit_checkpoint_functions_to_noise.py \
  --data-subdirs 0 001 003 005 01 w0 w025 w04 \
  --top-rank 1 --workers 2
```

The default test layout is
`<data-root>/<subdir>/<problem_id>/<dataset_index>.csv`, with indices `0`–`7`.
The default root is `synthetic_data/test_data/` relative to the repository.
Use `--data-template` for a different layout. This evaluator matches independent
variables by their names in the exported function signature and records MSE,
NMSE, and R² after fitting parameters independently to each CSV.

| Utility | Purpose |
| --- | --- |
| `collect_top_samples_all_problems.py` | Select per-problem JSON samples by a metric and convert functions to symbolic formulas. |
| `json_folder_to_excel.py` | Export the highest-scoring JSON records to Excel. |
| `merge_generation_and_top1_formulas.py` | Join selected formulas with benchmark metadata and generating formulas. |
| `calculate_generation_formula_complexity.py` | Add symbolic complexity values to a workbook. |
| `merge_statistics.py` | Combine similarity annotations, training statistics, and robustness metrics. |
| `plot_llm_mdsr.py` | Produce benchmark comparison figures and CSV summaries. |

Use each utility's `--help` to override its default paths. The plotting script
reads the shared log directory by default and expects
prepared `LLM_MDSR_000.xlsx`, `LLM_MDSR_001.xlsx`, and `LLM_MDSR_003.xlsx`
workbooks, including structural-similarity annotations. These are analysis
inputs rather than direct outputs of `main.py`.

## Repository layout

```text
LLM-MDSR/
├── main.py                    multi-problem search runner
├── paths.py                   shared repository-relative path defaults
├── generate_prompts.py        workbook-to-prompt generation
├── llmsr/                     sampling, evaluation, buffer, profiling
├── llm_engine/                optional local model server
├── data/                      example problem data and prompts
├── specs/                     original LLM-SR prompt specifications
├── logs/                      existing experiment artifacts
├── backup/                    archived experiment artifacts
├── tests/                     profiler regression test
└── *.py                       export, fitting, and reporting utilities
```

## Reference

Upstream code and documentation:
**[deep-symbolic-mathematics/LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR)**.
Its original license notice is retained in [LICENSE.upstream](LICENSE.upstream).
Existing per-file copyright and license headers are preserved.

The upstream repository provides this citation:

```bibtex
@article{shojaee2024llm,
  title={Llm-sr: Scientific equation discovery via programming with large language models},
  author={Shojaee, Parshin and Meidani, Kazem and Gupta, Shashank and Farimani, Amir Barati and Reddy, Chandan K},
  journal={arXiv preprint arXiv:2404.18400},
  year={2024}
}
```
