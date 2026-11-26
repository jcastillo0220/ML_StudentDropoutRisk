"""
Student Dropout Risk - Reproducible ML Pipeline.

Run from the repository root:
    python run_pipeline.py
"""

import os
import sys
import time
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXECUTED = RESULTS / "executed_notebooks"

RESULTS.mkdir(exist_ok=True)
EXECUTED.mkdir(exist_ok=True)

os.chdir(ROOT)

ENV = os.environ.copy()
ENV["MPLBACKEND"] = "Agg"
ENV["PYTHONUNBUFFERED"] = "1"

def run_step(title, command):

    print("\n" + "=" * 65, flush=True)
    print(title, flush=True)
    print("=" * 65, flush=True)

    start = time.perf_counter()

    subprocess.run(
        command,
        cwd=ROOT,
        env=ENV,
        check=True
    )

    elapsed = time.perf_counter() - start

    print(
        f"\nPASS: {title} ({elapsed:.2f}s)",
        flush=True
    )

def run_notebook(filename):

    run_step(
        f"Executing {filename}",
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to", "notebook",
            "--execute",
            str(ROOT / "notebooks" / filename),
            "--ExecutePreprocessor.kernel_name=python3",
            "--ExecutePreprocessor.timeout=600",
            f"--output-dir={EXECUTED}"
        ]
    )

def run_script(filename):

    run_step(
        f"Executing {filename}",
        [sys.executable, str(ROOT / filename)]
    )

def validate_results():

    required = [
        "full_dataset.csv",
        "processed_data.pkl",
        "baseline_comparison.csv",
        "tuned_comparison.csv",
        "best_hyperparameters.json",
        "early_prediction_comparison.csv",
        "scenario_features.json",
        "class_performance.csv",
        "subgroup_performance.csv",
        "dropout_error_summary.csv",
        "threshold_analysis.csv"
    ]

    for filename in required:

        path = RESULTS / filename

        assert path.is_file(), (
            f"Missing result: {filename}"
        )

        assert path.stat().st_size > 0, (
            f"Empty result: {filename}"
        )

        print(f"PASS: {filename}")

    baseline = pd.read_csv(
        RESULTS / "baseline_comparison.csv"
    )

    tuned = pd.read_csv(
        RESULTS / "tuned_comparison.csv"
    )

    early = pd.read_csv(
        RESULTS / "early_prediction_comparison.csv"
    )

    assert len(baseline) == 3
    assert len(tuned) == 3
    assert len(early) == 9

    charts = list(RESULTS.rglob("*.png"))

    assert len(charts) >= 15

    print("\nPASS: Three baseline models.")
    print("PASS: Three optimized models.")
    print("PASS: Nine early prediction experiments.")
    print(f"PASS: {len(charts)} charts available.")

def main():

    start = time.perf_counter()

    print("=" * 65)
    print("STUDENT DROPOUT RISK")
    print("COMPLETE MACHINE LEARNING PIPELINE")
    print("=" * 65)

    # Data preparation
    run_notebook("01_data_imports.ipynb")
    run_notebook("02_preprocessing.ipynb")

    # Individual model notebooks
    run_notebook("03_logistic_regression.ipynb")
    run_notebook("04_random_forest.ipynb")
    run_notebook("05_xgboost.ipynb")

    # Standardized experiments
    run_script("evaluate_baselines.py")
    run_script("tune_models.py")
    run_script("early_prediction.py")
    run_script("error_analysis.py")

    # Final result notebooks
    run_notebook("06_error_analysis.ipynb")
    run_notebook("07_final_results.ipynb")

    print("\n" + "=" * 65)
    print("FINAL VALIDATION")
    print("=" * 65)

    validate_results()

    elapsed = time.perf_counter() - start

    print("\n" + "=" * 65)
    print("PASS: COMPLETE PIPELINE SUCCESSFUL")
    print(f"Total execution time: {elapsed:.2f}s")
    print("=" * 65)

if __name__ == "__main__":
    main()
