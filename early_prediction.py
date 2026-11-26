import json
import pickle
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    make_scorer
)

from xgboost import XGBClassifier

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

with open(RESULTS / "processed_data.pkl", "rb") as file:
    data = pickle.load(file)

with open(RESULTS / "best_hyperparameters.json") as file:
    best_params = json.load(file)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

dropout_label = int(
    encoder.transform(["Dropout"])[0]
)

def dropout_recall(y_true, y_pred):
    return recall_score(
        y_true == dropout_label,
        y_pred == dropout_label,
        zero_division=0
    )

# Identify semester-dependent features
semester_1 = [
    col for col in X_train.columns
    if "1st sem" in col.lower()
]

semester_2 = [
    col for col in X_train.columns
    if "2nd sem" in col.lower()
]

# Conservatively exclude fields whose
# measurement dates may be ambiguous.
financial_status = [
    col for col in X_train.columns
    if col in [
        "Debtor",
        "Tuition fees up to date"
    ]
]

scenarios = {
    "Enrollment": [
        col for col in X_train.columns
        if col not in (
            semester_1
            + semester_2
            + financial_status
        )
    ],
    "First Semester": [
        col for col in X_train.columns
        if col not in (
            semester_2
            + financial_status
        )
    ],
    "Full Dataset": list(X_train.columns)
}

with open(
    RESULTS / "scenario_features.json", "w"
) as file:
    json.dump(scenarios, file, indent=4)

def build_models():

    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            random_state=42
        ))
    ])

    lr.set_params(
        **best_params["Logistic Regression"]
    )

    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=1,
        **best_params["Random Forest"]
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        **best_params["XGBoost"]
    )

    return {
        "Logistic Regression": lr,
        "Random Forest": rf,
        "XGBoost": xgb
    }

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = {
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "dropout_recall": make_scorer(
        dropout_recall
    )
}

all_results = []

for scenario, features in scenarios.items():

    print("\n" + "=" * 65)
    print("SCENARIO:", scenario)
    print("FEATURE COUNT:", len(features))
    print("=" * 65)

    train = X_train[features]
    test = X_test[features]

    for name, model in build_models().items():

        print("\nEvaluating:", name)

        cv_results = cross_validate(
            model,
            train,
            y_train_encoded,
            cv=cv,
            scoring=scoring,
            n_jobs=4,
            error_score="raise"
        )

        start = time.perf_counter()

        model.fit(train, y_train_encoded)

        training_time = (
            time.perf_counter() - start
        )

        predicted_encoded = model.predict(test)

        predicted = encoder.inverse_transform(
            predicted_encoded
        )

        probabilities = model.predict_proba(test)

        actual_dropout = y_test == "Dropout"
        predicted_dropout = predicted == "Dropout"

        metrics = {
            "Scenario": scenario,
            "Model": name,
            "Features": len(features),
            "CV Macro F1": cv_results[
                "test_macro_f1"
            ].mean(),
            "CV Macro F1 Std": cv_results[
                "test_macro_f1"
            ].std(),
            "CV Dropout Recall": cv_results[
                "test_dropout_recall"
            ].mean(),
            "Accuracy": accuracy_score(
                y_test, predicted
            ),
            "Macro Precision": precision_score(
                y_test, predicted,
                average="macro",
                zero_division=0
            ),
            "Macro Recall": recall_score(
                y_test, predicted,
                average="macro",
                zero_division=0
            ),
            "Macro F1": f1_score(
                y_test, predicted,
                average="macro",
                zero_division=0
            ),
            "Dropout Precision": precision_score(
                actual_dropout,
                predicted_dropout,
                zero_division=0
            ),
            "Dropout Recall": recall_score(
                actual_dropout,
                predicted_dropout,
                zero_division=0
            ),
            "Dropout F1": f1_score(
                actual_dropout,
                predicted_dropout,
                zero_division=0
            ),
            "ROC-AUC": roc_auc_score(
                y_test_encoded,
                probabilities,
                multi_class="ovr",
                average="macro"
            ),
            "Training Time (s)": training_time
        }

        all_results.append(metrics)

        scenario_slug = (
            scenario.lower().replace(" ", "_")
        )

        model_slug = (
            name.lower().replace(" ", "_")
        )

        predictions = pd.DataFrame({
            "Student_Index": X_test.index.to_numpy(),
            "Actual": y_test.to_numpy(),
            "Predicted": predicted,
            "Correct": y_test.to_numpy() == predicted
        })

        for index, class_name in enumerate(
            encoder.classes_
        ):
            predictions[
                f"Probability_{class_name}"
            ] = probabilities[:, index]

        predictions.to_csv(
            RESULTS / (
                f"{scenario_slug}_"
                f"{model_slug}_predictions.csv"
            ),
            index=False
        )

        print(
            f"CV Macro F1: "
            f"{metrics['CV Macro F1']:.4f}"
        )

        print(
            f"Test Accuracy: "
            f"{metrics['Accuracy']:.4f}"
        )

        print(
            f"Test Macro F1: "
            f"{metrics['Macro F1']:.4f}"
        )

        print(
            f"Dropout Recall: "
            f"{metrics['Dropout Recall']:.4f}"
        )

comparison = pd.DataFrame(all_results)

comparison.to_csv(
    RESULTS / "early_prediction_comparison.csv",
    index=False
)

# Generate comparison charts
scenario_order = list(scenarios.keys())

for metric in [
    "Accuracy",
    "Macro F1",
    "Dropout Recall",
    "ROC-AUC"
]:

    pivot = comparison.pivot(
        index="Scenario",
        columns="Model",
        values=metric
    ).reindex(scenario_order)

    ax = pivot.plot(
        kind="bar",
        figsize=(10, 6)
    )

    ax.set_title(
        f"{metric} by Prediction Scenario"
    )

    ax.set_ylabel(metric)
    ax.set_xlabel("Prediction Scenario")
    ax.set_ylim(0, 1)
    ax.legend(title="Model")

    plt.xticks(rotation=0)
    plt.tight_layout()

    filename = (
        metric.lower()
        .replace(" ", "_")
        .replace("-", "_")
    )

    plt.savefig(
        RESULTS / f"early_prediction_{filename}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

print("\n" + "=" * 65)
print("EARLY PREDICTION RESULTS")
print("=" * 65)

columns = [
    "Scenario",
    "Model",
    "Features",
    "CV Macro F1",
    "Accuracy",
    "Macro F1",
    "Dropout Recall",
    "ROC-AUC"
]

print(
    comparison[columns]
    .round(4)
    .to_string(index=False)
)

print("\nPASS: All nine experiments completed.")
print("PASS: Scenario features saved.")
print("PASS: Predictions saved.")
print("PASS: Comparison charts generated.")
