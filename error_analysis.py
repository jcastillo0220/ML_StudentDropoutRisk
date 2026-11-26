import pickle
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve
)

RESULTS = Path("results")
CHARTS = RESULTS / "charts"
CHARTS.mkdir(parents=True, exist_ok=True)

with open(RESULTS / "processed_data.pkl", "rb") as file:
    data = pickle.load(file)

X_test = data["X_test"].copy()

assert X_test.index.is_unique

X_test["Student_Index"] = X_test.index

classes = ["Dropout", "Enrolled", "Graduate"]

models = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost"
}

groups = [
    "Gender",
    "Scholarship holder",
    "Debtor",
    "Tuition fees up to date",
    "International",
    "Educational special needs"
]

class_results = []
subgroup_results = []
error_results = []

for name, slug in models.items():

    for version in ["baseline", "tuned"]:

        print("\n" + "=" * 65)
        print(name, "-", version.upper())
        print("=" * 65)

        predictions = pd.read_csv(
            RESULTS / f"{slug}_{version}_predictions.csv"
        )

        assert len(predictions) == len(X_test)
        assert predictions["Student_Index"].is_unique

        df = predictions.merge(
            X_test.reset_index(drop=True),
            on="Student_Index",
            how="left",
            validate="one_to_one"
        )

        actual = df["Actual"]
        predicted = df["Predicted"]

        # Class-level performance
        report = classification_report(
            actual,
            predicted,
            labels=classes,
            output_dict=True,
            zero_division=0
        )

        for label in classes:

            class_results.append({
                "Model": name,
                "Version": version,
                "Class": label,
                "Precision": report[label]["precision"],
                "Recall": report[label]["recall"],
                "F1": report[label]["f1-score"],
                "Support": report[label]["support"]
            })

        # Confusion matrix
        matrix = confusion_matrix(
            actual,
            predicted,
            labels=classes
        )

        fig, ax = plt.subplots(figsize=(7, 6))

        ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=classes
        ).plot(
            ax=ax,
            colorbar=False
        )

        ax.set_title(
            f"{name} - {version.title()}"
        )

        fig.tight_layout()

        fig.savefig(
            CHARTS / (
                f"{slug}_{version}_confusion_matrix.png"
            ),
            dpi=200
        )

        plt.close(fig)

        # Dropout detection errors
        actual_dropout = actual == "Dropout"
        predicted_dropout = predicted == "Dropout"

        tp = int(
            (actual_dropout & predicted_dropout).sum()
        )

        fn = int(
            (actual_dropout & ~predicted_dropout).sum()
        )

        fp = int(
            (~actual_dropout & predicted_dropout).sum()
        )

        tn = int(
            (~actual_dropout & ~predicted_dropout).sum()
        )

        error_results.append({
            "Model": name,
            "Version": version,
            "True Positives": tp,
            "False Negatives": fn,
            "False Positives": fp,
            "True Negatives": tn
        })

        print("Correctly identified dropouts:", tp)
        print("Missed dropouts:", fn)
        print("False alarms:", fp)

        # Save missed dropout records
        missed = df[
            actual_dropout & ~predicted_dropout
        ].copy()

        missed.to_csv(
            RESULTS / f"{slug}_{version}_missed_dropouts.csv",
            index=False
        )

        # Subgroup evaluation
        for feature in groups:

            if feature not in df.columns:
                continue

            for value, subset in df.groupby(feature):

                a = subset["Actual"]
                p = subset["Predicted"]

                a_dropout = a == "Dropout"
                p_dropout = p == "Dropout"

                dropout_count = int(a_dropout.sum())
                non_dropout_count = int(
                    (~a_dropout).sum()
                )

                false_positives = int(
                    (~a_dropout & p_dropout).sum()
                )

                subgroup_results.append({
                    "Model": name,
                    "Version": version,
                    "Feature": feature,
                    "Group": value,
                    "Students": len(subset),
                    "Actual Dropouts": dropout_count,
                    "Accuracy": accuracy_score(a, p),
                    "Dropout Recall": (
                        recall_score(
                            a_dropout,
                            p_dropout,
                            zero_division=0
                        )
                        if dropout_count > 0
                        else np.nan
                    ),
                    "Dropout Precision": (
                        precision_score(
                            a_dropout,
                            p_dropout,
                            zero_division=0
                        )
                        if p_dropout.sum() > 0
                        else np.nan
                    ),
                    "False Positive Rate": (
                        false_positives / non_dropout_count
                        if non_dropout_count > 0
                        else np.nan
                    ),
                    "Small Sample": (
                        len(subset) < 30
                        or dropout_count < 20
                        or non_dropout_count < 20
                    )
                })

        print("PASS: Error analysis completed.")

pd.DataFrame(class_results).to_csv(
    RESULTS / "class_performance.csv",
    index=False
)

subgroups = pd.DataFrame(subgroup_results)

subgroups.to_csv(
    RESULTS / "subgroup_performance.csv",
    index=False
)

errors = pd.DataFrame(error_results)

errors.to_csv(
    RESULTS / "dropout_error_summary.csv",
    index=False
)

# Baseline versus tuned performance
baseline = pd.read_csv(
    RESULTS / "baseline_comparison.csv"
)

tuned = pd.read_csv(
    RESULTS / "tuned_comparison.csv"
)

for metric in [
    "Accuracy",
    "Macro F1",
    "Dropout Recall",
    "ROC-AUC"
]:

    comparison = baseline[
        ["Model", metric]
    ].merge(
        tuned[["Model", metric]],
        on="Model",
        suffixes=("_Baseline", "_Tuned")
    )

    comparison = comparison.set_index("Model")

    comparison.columns = ["Baseline", "Tuned"]

    ax = comparison.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        f"Baseline vs Tuned - {metric}"
    )

    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)

    plt.xticks(rotation=0)
    plt.tight_layout()

    filename = (
        metric.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        CHARTS / f"comparison_{filename}.png",
        dpi=200
    )

    plt.close()

# Exploratory first-semester threshold analysis
threshold_results = []

fig, ax = plt.subplots(figsize=(9, 6))

for name, slug in models.items():

    predictions = pd.read_csv(
        RESULTS / (
            f"first_semester_{slug}_predictions.csv"
        )
    )

    actual = predictions["Actual"] == "Dropout"

    probabilities = predictions[
        "Probability_Dropout"
    ]

    precision, recall, _ = precision_recall_curve(
        actual,
        probabilities
    )

    ax.plot(
        recall,
        precision,
        label=name
    )

    for threshold in np.arange(0.1, 1.0, 0.05):

        flagged = probabilities >= threshold

        threshold_results.append({
            "Model": name,
            "Threshold": round(float(threshold), 2),
            "Students Flagged": int(flagged.sum()),
            "True Positives": int(
                (actual & flagged).sum()
            ),
            "False Positives": int(
                (~actual & flagged).sum()
            ),
            "Missed Dropouts": int(
                (actual & ~flagged).sum()
            ),
            "Precision": precision_score(
                actual,
                flagged,
                zero_division=0
            ),
            "Recall": recall_score(
                actual,
                flagged,
                zero_division=0
            ),
            "F1": f1_score(
                actual,
                flagged,
                zero_division=0
            )
        })

ax.set_title(
    "First-Semester Dropout Precision-Recall"
)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend()

fig.tight_layout()

fig.savefig(
    CHARTS / "first_semester_precision_recall.png",
    dpi=200
)

plt.close(fig)

pd.DataFrame(threshold_results).to_csv(
    RESULTS / "threshold_analysis.csv",
    index=False
)

print("\n" + "=" * 65)
print("DROPOUT ERROR SUMMARY")
print("=" * 65)

print(errors.to_string(index=False))

print("\nPASS: Class performance saved.")
print("PASS: Subgroup performance saved.")
print("PASS: Dropout error summary saved.")
print("PASS: Threshold analysis saved.")
print("PASS: All charts generated.")
