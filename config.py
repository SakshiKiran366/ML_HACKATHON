from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATASET_ROOT / "train"
TEST_DIR = DATASET_ROOT / "student_test"

OUTPUT_DIR = PROJECT_ROOT / "output"
REPORT_DIR = OUTPUT_DIR / "reports"
PROCESSED_DIR = OUTPUT_DIR / "processed"
LOG_DIR = OUTPUT_DIR / "logs"
