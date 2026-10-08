"""Default paths resolved from the repository, independent of the working directory."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT / "synthetic_data" / "PhysicsMDRS_Synthetic_Dataset"
CSV_ROOT = PROJECT_ROOT / "physicsMDSR_Range_CSV"
LOGS_ROOT = CSV_ROOT / "logs"
TEST_DATA_ROOT = REPO_ROOT / "synthetic_data" / "test_data"
BENCHMARK_FILE = REPO_ROOT / "MSSR" / "physicsMDSR_Range.xlsx"
