import pickle
import time
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

from xgboost import XGBClassifier

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

# Load preprocessed data
with open(RESULTS / "processed_data.pkl", "rb") as file:
    data = pickle.load(file)

X_train = data["X_train"]
X_test = data["X_test"]
X_train_scaled = data["X_train_scaled"]
X_test_scaled = data["X_test_scaled"]
y_train = data["y_train"]
y_test = data["y_test"]

# Encode target classes
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

print("=" * 65)
print("BASELINE MODEL EVALUATION")
print("=" * 65)
print("Training samples:", len(y_train))
print("Testing samples:", len(y_test))
print("Classes:", list(encoder.classes_))

# Define baseline models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=5000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )
}

all_metrics = []

for name, model in models.items():

    print("\n" + "=" * 65)
    print(name.upper())
    print("=" * 65)

    if name == "Logistic Regression":
        train_features = X_train_scaled
        test_features = X_test_scaled
    else:
        train_features = X_train
        test_features = X_test

    start = time.perf_counter()

    model.fit(train_features, y_train_encoded)

    training_time = time.perf_counter() - start

    start = time.perf_counter()

    predicted_encoded = model.predict(test_features)
    probabilities = model.predict_proba(test_features)

    prediction_time = time.perf_counter() - start

    predicted = encoder.inverse_transform(
        predicted_encoded
    )

    actual_dropout = y_test == "Dropout"
    predicted_dropout = predicted == "Dropout"

    metrics = {
        "Model": name,
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
        "Weighted F1": f1_score(
            y_test, predicted,
            average="weighted",
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
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time
    }

    all_metrics.append(metrics)

    print(classification_report(
        y_test,
        predicted,
        zero_division=0
    ))

    for key, value in metrics.items():
        if key != "Model":
            print(f"{key}: {value:.6f}")

    # Save predictions for error analysis
    predictions = pd.DataFrame({
        "Student_Index": X_test.index.to_numpy(),
        "Actual": y_test.to_numpy(),
        "Predicted": predicted,
        "Correct": y_test.to_numpy() == predicted
    })

    for index, class_name in enumerate(encoder.classes_):
        predictions[f"Probability_{class_name}"] = (
            probabilities[:, index]
        )

    filename = (
        name.lower().replace(" ", "_")
        + "_baseline_predictions.csv"
    )

    predictions.to_csv(
        RESULTS / filename,
        index=False
    )

    print(f"PASS: Saved {filename}")

# Save consolidated comparison
comparison = pd.DataFrame(all_metrics)

comparison = comparison.sort_values(
    "Macro F1",
    ascending=False
)

comparison.to_csv(
    RESULTS / "baseline_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("FINAL BASELINE COMPARISON")
print("=" * 65)

print(comparison.round(4).to_string(index=False))

print("\nPASS: All baseline models evaluated.")
print("PASS: Predictions saved.")
print("PASS: Model comparison saved.")
