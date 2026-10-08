# MDSR

MDSR brings together symbolic-regression implementations for discovering one
equation structure across several related datasets, with numerical parameters
fitted separately for each dataset. It includes PySR-based discovery,
LLM-based discovery, and the original multi-stage selection experiments.

The root `setup.sh` and `run.sh` scripts serve **MDSR-PySR**. **LLM-MDSR** has
its own Python runner and environment. The two original implementations are
pinned as Git submodules; `LLM-MDSR/` is a regular directory.

## Components

| Component | Status | Purpose |
| --- | --- | --- |
| [`MDSR-PySR`](MDSR-PySR/) | Supported | Modular data preparation, PySR discovery, shared-formula evaluation, robustness analysis, and reporting. |
| [`LLM-MDSR`](LLM-MDSR/README.md) | Experimental | LLM-generated equation functions, multi-dataset fitting, checkpoint recovery, and noise/distribution-shift analysis. |
| [`Multi-stage-Selection-Symbolic-Regression`](Multi-stage-Selection-Symbolic-Regression/) | Legacy/reference | Original MSSR experiments and datasets. Its current script contains Windows-specific paths and is not launched by the root runner. |

**LLM-MDSR is adapted from
[LLM-SR](https://github.com/deep-symbolic-mathematics/LLM-SR)**, the official
implementation of
[LLM-SR: Scientific Equation Discovery via Programming with Large Language Models](https://arxiv.org/abs/2404.18400).
Its contribution here is shared-formula discovery and evaluation across related
physics datasets. See the [LLM-MDSR README](LLM-MDSR/README.md) for setup,
implementation details, and the upstream citation.

Some component documentation uses **MSSR** for the multi-stage selection method;
the umbrella repository and primary implementation are named **MDSR**.

## PySR quick start

Clone the repository together with its pinned submodules:

```bash
git clone --recurse-submodules https://github.com/daniel2478178/MDSR.git
cd MDSR
./setup.sh
conda activate mdsr-pysr
```

Check the complete pipeline without starting a long experiment:

```bash
./run.sh all --mode xonly --dry-run
```

Run the core workflow:

```bash
./run.sh core --mode xonly
```

`setup.sh` initializes the submodules and creates or updates the Conda
environment declared in `MDSR-PySR/environment.yml`. Use
`./setup.sh --skip-env` when only the submodules need to be initialized.

For an existing clone whose component directories are empty, run:

```bash
git submodule update --init --recursive
```

## LLM-MDSR quick start

Follow the [LLM-MDSR setup guide](LLM-MDSR/README.md) to create its Conda
environment, prepare the datasets and prompts, and set `API_KEY` for the
configured API provider. Then run from `LLM-MDSR/`:

```bash
python main.py --api_model YOUR_MODEL --num_processes 1
```

The current runner processes `P01`–`P59`, using eight datasets per problem and
a default budget of 12 generated samples. Its defaults resolve from the
repository location and use the existing
`synthetic_data/PhysicsMDRS_Synthetic_Dataset/` layout, including `train_data/`.
Shared defaults live in `LLM-MDSR/paths.py`; no machine-specific absolute paths
are required. Training CSVs and prompts are included in the repository and
available when cloning. Root `setup.sh` does not create the LLM environment,
and root `run.sh` does not launch this component.

## PySR pipeline commands

`run.sh` forwards every option to the primary pipeline. With no arguments it
shows the pipeline help.

| Command | Purpose |
| --- | --- |
| `prepare` | Generate the base benchmark datasets. |
| `discover` | Discover candidate expressions with PySR. |
| `evaluate` | Fit generalized candidates across dataset groups. |
| `merge` | Rank and merge non-redundant shared formulas. |
| `robustness` | Evaluate distribution shift and target noise. |
| `report` | Generate plots from existing evaluation results. |
| `core` | Run `prepare -> discover -> evaluate -> merge`. |
| `all` | Run every stage. |

Examples:

```bash
# Inspect all options
./run.sh --help

# Run a single stage
./run.sh report

# Include physical parameters and constants during discovery
./run.sh discover --mode with-params --iterations 3

# Use a conservative process profile on a laptop
./run.sh core --outer-processes 2 --pysr-processes 4 --workers 2

# Select a different Python interpreter
./run.sh core --python /path/to/python
```

PySR discovery is the expensive stage and uses CPU processes. Start with the
conservative profile above before increasing process counts on a larger server.

## PySR environment options

Conda is recommended because PySR manages a Julia/SymbolicRegression backend:

```bash
conda env create -f MDSR-PySR/environment.yml
conda activate mdsr-pysr
```

For analysis, data preparation, plotting, and tests without PySR:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r MDSR-PySR/requirements.txt
```

Install `MDSR-PySR/requirements-pysr.txt` instead when discovery is required.
You may also set `PYTHON_BIN=/path/to/python` when invoking `run.sh`.

## PySR data and outputs

Default inputs, generated data, and results live inside `MDSR-PySR/`:

```text
MDSR-PySR/
├── physicsMDSR_Range.xlsx       benchmark metadata
├── physicsMDSR_Range_CSV/       generated base datasets
├── generated/                   robustness datasets
└── results/                     merged metrics and plots
```

Most paths can be replaced with command-line options. Existing base data and
robustness datasets are reused by default; pass `--force` only when they should
be regenerated. See the
[`MDSR-PySR` documentation](MDSR-PySR/README.md) for the workbook schema,
stage-level interfaces, output details, and resume behavior.

LLM-MDSR writes checkpoints, sample records, and TensorBoard events beneath its
configured dataset root. Its [README](LLM-MDSR/README.md) describes the input
layout, output paths, resume behavior, and analysis commands.

## PySR verification

Run the lightweight checks from the repository root:

```bash
./run.sh all --mode xonly --dry-run
(
  cd MDSR-PySR
  python -m unittest discover -v
)
```

The GitHub Actions workflow runs the same unit and dry-run checks on Python
3.11. It intentionally does not start a full PySR/Julia search.

## Repository structure

```text
.
├── MDSR-PySR/                              supported pipeline (submodule)
├── LLM-MDSR/                               experimental LLM-based implementation
├── Multi-stage-Selection-Symbolic-Regression/ legacy implementation (submodule)
├── synthetic_data/                         included synthetic benchmark data and prompts
├── .github/workflows/ci.yml                 lightweight automated checks
├── run.sh                                   root pipeline launcher
└── setup.sh                                 submodule and Conda setup
```

The root repository records exact commits for its two Git submodules. Changes
inside those submodules must first be committed and pushed in their own
repositories; then commit the updated submodule pointer here. Files in regular
directories such as `LLM-MDSR/` are tracked directly by the root repository.
