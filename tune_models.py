import json
import pickle
import time
from pathlib import Path

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    make_scorer
)

from xgboost import XGBClassifier

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

with open(RESULTS / "processed_data.pkl", "rb") as file:
    data = pickle.load(file)

X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]

encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

dropout_label = int(encoder.transform(["Dropout"])[0])

def dropout_recall(y_true, y_pred):
    return recall_score(
        y_true == dropout_label,
        y_pred == dropout_label,
        zero_division=0
    )

scoring = {
    "macro_f1": "f1_macro",
    "accuracy": "accuracy",
    "dropout_recall": make_scorer(dropout_recall)
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        random_state=42
    ))
])

lr_params = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__class_weight": [None, "balanced"]
}

rf_model = RandomForestClassifier(
    random_state=42,
    n_jobs=1
)

rf_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", 0.5],
    "class_weight": [None, "balanced"]
}

xgb_model = XGBClassifier(
    objective="multi:softprob",
    num_class=len(encoder.classes_),
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=1
)

xgb_params = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 2, 5]
}

searches = {
    "Logistic Regression": GridSearchCV(
        lr_pipeline,
        lr_params,
        scoring=scoring,
        refit="macro_f1",
        cv=cv,
        n_jobs=4,
        verbose=1,
        error_score="raise"
    ),
    "Random Forest": RandomizedSearchCV(
        rf_model,
        rf_params,
        n_iter=25,
        scoring=scoring,
        refit="macro_f1",
        cv=cv,
        n_jobs=4,
        random_state=42,
        verbose=1,
        error_score="raise"
    ),
    "XGBoost": RandomizedSearchCV(
        xgb_model,
        xgb_params,
        n_iter=30,
        scoring=scoring,
        refit="macro_f1",
        cv=cv,
        n_jobs=4,
        random_state=42,
        verbose=1,
        error_score="raise"
    )
}

all_metrics = []
all_parameters = {}

for name, search in searches.items():

    print("\n" + "=" * 65)
    print("OPTIMIZING:", name)
    print("=" * 65)

    slug = name.lower().replace(" ", "_")

    start = time.perf_counter()

    search.fit(X_train, y_train_encoded)

    tuning_time = time.perf_counter() - start

    model = search.best_estimator_

    print("\nBest parameters:")
    print(search.best_params_)

    print("Best CV Macro F1:", search.best_score_)

    predicted_encoded = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    predicted = encoder.inverse_transform(
        predicted_encoded
    )

    actual_dropout = y_test == "Dropout"
    predicted_dropout = predicted == "Dropout"

    metrics = {
        "Model": name,
        "CV Macro F1": search.best_score_,
        "CV Dropout Recall": search.cv_results_[
            "mean_test_dropout_recall"
        ][search.best_index_],
        "Accuracy": accuracy_score(y_test, predicted),
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
        "Tuning Time (s)": tuning_time
    }

    all_metrics.append(metrics)
    all_parameters[name] = search.best_params_

    print("\nTest classification report:")
    print(classification_report(
        y_test,
        predicted,
        zero_division=0
    ))

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

    predictions.to_csv(
        RESULTS / f"{slug}_tuned_predictions.csv",
        index=False
    )

    pd.DataFrame(search.cv_results_).to_csv(
        RESULTS / f"{slug}_cv_results.csv",
        index=False
    )

    with open(
        RESULTS / f"{slug}_tuned_model.pkl",
        "wb"
    ) as file:
        pickle.dump({
            "model": model,
            "encoder": encoder
        }, file)

    print(f"PASS: {name} optimization complete.")

comparison = pd.DataFrame(all_metrics)

comparison = comparison.sort_values(
    "CV Macro F1",
    ascending=False
)

comparison.to_csv(
    RESULTS / "tuned_comparison.csv",
    index=False
)

with open(
    RESULTS / "best_hyperparameters.json",
    "w"
) as file:
    json.dump(
        all_parameters,
        file,
        indent=4,
        default=str
    )

print("\n" + "=" * 65)
print("FINAL OPTIMIZED MODEL COMPARISON")
print("=" * 65)

print(comparison.round(4).to_string(index=False))

print("\nPASS: All models optimized.")
print("PASS: Best hyperparameters saved.")
print("PASS: Tuned predictions saved.")
print("PASS: Tuned models saved.")
