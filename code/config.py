from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = ROOT / "figures"
TABLES_DIR = ROOT / "tables"
DATA_DIR = ROOT / "data"

RANDOM_SEED = 20260602
N_GRID = 180
N_MAIN = 250
N_REPS = 200

KERNEL_NAME = "gaussian"
BANDWIDTH_GRID = [0.035, 0.05, 0.07, 0.09, 0.12, 0.16, 0.22]
SPLINE_KNOT_GRID = [5, 7, 9, 12, 15]
SMOOTHING_GRID = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
N_FOLDS = 5

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
