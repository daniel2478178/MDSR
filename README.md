# MDSR

MDSR is the top-level repository for two related symbolic-regression projects.
It pins each project as a Git submodule so that experiments remain reproducible
while the implementations can evolve independently.

The supported entry point is **MDSR-PySR**, a staged pipeline for discovering
and evaluating one symbolic formula across several related physics datasets.

## Components

| Component | Status | Purpose |
| --- | --- | --- |
| [`MDSR-PySR`](MDSR-PySR/) | Supported | Modular data preparation, PySR discovery, shared-formula evaluation, robustness analysis, and reporting. |
| [`Multi-stage-Selection-Symbolic-Regression`](Multi-stage-Selection-Symbolic-Regression/) | Legacy/reference | Original MSSR experiments and datasets. Its current script contains Windows-specific paths and is not launched by the root runner. |

Some component documentation uses **MSSR** for the multi-stage selection method;
the umbrella repository and primary implementation are named **MDSR**.

## Quick start

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

## Pipeline commands

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

## Environment options

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

## Data and outputs

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

## Verification

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
├── Multi-stage-Selection-Symbolic-Regression/ legacy implementation (submodule)
├── .github/workflows/ci.yml                 lightweight automated checks
├── run.sh                                   root pipeline launcher
└── setup.sh                                 submodule and Conda setup
```

The root repository records exact submodule commits. Changes inside a component
must first be committed and pushed in that component's repository; then commit
the updated submodule pointer here. This prevents an umbrella commit from
silently depending on uncommitted child changes.
