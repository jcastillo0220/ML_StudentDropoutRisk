"""
Student Success Analytics Dashboard.

Machine Learning Term Project:
Learning for Breaking Barriers.

Launch with:
    python -m streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
CHARTS = RESULTS / "charts"

st.set_page_config(
    page_title="Student Success Analytics",
    page_icon="🎓",
    layout="wide"
)

@st.cache_data
def load_csv(filename):
    return pd.read_csv(RESULTS / filename)

required = [
    "baseline_comparison.csv",
    "tuned_comparison.csv",
    "early_prediction_comparison.csv",
    "subgroup_performance.csv",
    "threshold_analysis.csv",
    "dropout_error_summary.csv",
    "class_performance.csv"
]

missing = [
    name for name in required
    if not (RESULTS / name).exists()
]

if missing:
    st.error("Research results are missing.")
    st.write(missing)
    st.code("python run_pipeline.py")
    st.stop()

baseline = load_csv("baseline_comparison.csv")
tuned = load_csv("tuned_comparison.csv")
early = load_csv("early_prediction_comparison.csv")
subgroups = load_csv("subgroup_performance.csv")
thresholds = load_csv("threshold_analysis.csv")
errors = load_csv("dropout_error_summary.csv")
classes = load_csv("class_performance.csv")

st.sidebar.title("Student Success Analytics")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Model Comparison",
        "Early Prediction",
        "Error and Subgroup Analysis",
        "Intervention Simulator",
        "Limitations and Ethics"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "Machine Learning Term Project"
)

st.sidebar.caption(
    "Learning for Breaking Barriers"
)

st.sidebar.caption(
    "Research demonstration only."
)

def percent(value):
    return f"{value:.2%}"

def show_image(path):
    if path.exists():
        st.image(str(path))
    else:
        st.warning(f"Chart unavailable: {path.name}")

# =====================================================
# PROJECT OVERVIEW
# =====================================================

if page == "Project Overview":

    st.title("Student Success Analytics")

    st.subheader(
        "Breaking Educational Barriers "
        "Through Machine Learning"
    )

    st.write(
        "This project investigates whether machine "
        "learning can identify students at risk of "
        "dropping out and support earlier academic "
        "interventions."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Students", "4,424")
    c2.metric("Input features", 36)
    c3.metric("Model configurations", 6)
    c4.metric("Early experiments", len(early))

    st.divider()

    st.subheader("Dataset Distribution")

    distribution = pd.DataFrame({
        "Outcome": [
            "Graduate",
            "Dropout",
            "Enrolled"
        ],
        "Students": [2209, 1421, 794]
    })

    st.bar_chart(
        distribution.set_index("Outcome")
    )

    st.dataframe(
        distribution,
        hide_index=True
    )

    st.subheader("Research Questions")

    st.markdown("""
    - Can machine learning identify students
      at risk of dropping out?
    - Does hyperparameter optimization improve
      classification performance?
    - How early can dropout risk be identified?
    - Does performance differ across student groups?
    - How does intervention workload change with
      the dropout probability threshold?
    """)

    st.subheader("Data Source")

    st.markdown(
        "[UCI Predict Students' Dropout and "
        "Academic Success Dataset]"
        "(https://archive.ics.uci.edu/dataset/697/"
        "predict+students+dropout+and+academic+success)"
    )

# =====================================================
# MODEL COMPARISON
# =====================================================

elif page == "Model Comparison":

    st.title("Model Performance Comparison")

    st.write(
        "All six model configurations were evaluated "
        "on the same 885 test students."
    )

    metric = st.selectbox(
        "Evaluation metric",
        [
            "Macro F1",
            "Accuracy",
            "Dropout Recall",
            "Dropout Precision",
            "ROC-AUC"
        ]
    )

    comparison = baseline[
        ["Model", metric]
    ].merge(
        tuned[["Model", metric]],
        on="Model",
        suffixes=("_Baseline", "_Tuned")
    )

    comparison = comparison.set_index("Model")
    comparison.columns = ["Baseline", "Tuned"]

    st.bar_chart(comparison, stack=False)

    st.dataframe(
        comparison.style.format("{:.2%}")
    )

    st.subheader("Baseline Results")

    st.dataframe(
        baseline.round(4),
        hide_index=True
    )

    st.subheader("Optimized Results")

    st.dataframe(
        tuned.round(4),
        hide_index=True
    )

    st.info(
        "Tuned Random Forest achieved the highest "
        "macro F1. Baseline Logistic Regression "
        "achieved the highest full-information "
        "dropout recall."
    )

# =====================================================
# EARLY PREDICTION
# =====================================================

elif page == "Early Prediction":

    st.title("How Early Can We Predict Dropout?")

    st.write(
        "Three feature-availability scenarios "
        "were evaluated using all three algorithms."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Enrollment features", 22)
    c2.metric("First-semester features", 28)
    c3.metric("Full dataset features", 36)

    metric = st.selectbox(
        "Performance metric",
        [
            "Macro F1",
            "Accuracy",
            "Dropout Recall",
            "ROC-AUC",
            "CV Macro F1"
        ]
    )

    comparison = early.pivot(
        index="Scenario",
        columns="Model",
        values=metric
    )

    comparison = comparison.reindex([
        "Enrollment",
        "First Semester",
        "Full Dataset"
    ])

    st.line_chart(comparison)

    st.dataframe(
        comparison.style.format("{:.2%}")
    )

    st.subheader("All Early Prediction Results")

    st.dataframe(
        early.round(4),
        hide_index=True
    )

    st.success(
        "First-semester academic information "
        "substantially improved performance "
        "relative to enrollment information alone."
    )

    st.warning(
        "Feature measurement timing must be "
        "verified before prospective deployment."
    )

# =====================================================
# ERROR AND SUBGROUP ANALYSIS
# =====================================================

elif page == "Error and Subgroup Analysis":

    st.title("Error and Subgroup Analysis")

    c1, c2 = st.columns(2)

    model = c1.selectbox(
        "Model",
        [
            "Logistic Regression",
            "Random Forest",
            "XGBoost"
        ],
        index=1
    )

    version = c2.selectbox(
        "Configuration",
        ["baseline", "tuned"],
        index=1
    )

    selected = errors[
        (errors["Model"] == model)
        & (errors["Version"] == version)
    ].iloc[0]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Identified dropouts",
        int(selected["True Positives"])
    )

    c2.metric(
        "Missed dropouts",
        int(selected["False Negatives"])
    )

    c3.metric(
        "False alarms",
        int(selected["False Positives"])
    )

    slug = model.lower().replace(" ", "_")

    st.subheader("Confusion Matrix")

    show_image(
        CHARTS /
        f"{slug}_{version}_confusion_matrix.png"
    )

    st.subheader("Class Performance")

    selected_classes = classes[
        (classes["Model"] == model)
        & (classes["Version"] == version)
    ]

    st.dataframe(
        selected_classes.round(4),
        hide_index=True
    )

    st.subheader("Student Subgroup Performance")

    feature = st.selectbox(
        "Student characteristic",
        sorted(subgroups["Feature"].unique())
    )

    group = subgroups[
        (subgroups["Model"] == model)
        & (subgroups["Version"] == version)
        & (subgroups["Feature"] == feature)
    ].copy()

    group["Group"] = group["Group"].astype(str)

    st.bar_chart(
        group.set_index("Group")[
            [
                "Accuracy",
                "Dropout Recall",
                "False Positive Rate"
            ]
        ],
        stack=False
    )

    st.dataframe(
        group.round(4),
        hide_index=True
    )

    if group["Small Sample"].any():
        st.warning(
            "At least one subgroup has limited "
            "sample support. Interpret its "
            "performance cautiously."
        )

    st.caption(
        "Group labels use the original dataset "
        "codes. Performance differences do not "
        "establish causation or discrimination."
    )

# =====================================================
# INTERVENTION SIMULATOR
# =====================================================

elif page == "Intervention Simulator":

    st.title("Academic Intervention Simulator")

    st.write(
        "Explore how first-semester dropout "
        "probability thresholds affect intervention "
        "workload and detection performance."
    )

    model = st.selectbox(
        "Model",
        sorted(thresholds["Model"].unique()),
        index=2
    )

    model_data = thresholds[
        thresholds["Model"] == model
    ].sort_values("Threshold")

    available = model_data[
        "Threshold"
    ].round(2).tolist()

    threshold = st.select_slider(
        "Dropout probability threshold",
        options=available,
        value=0.30
    )

    selected = model_data[
        model_data["Threshold"].round(2)
        == threshold
    ].iloc[0]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Students flagged",
        int(selected["Students Flagged"])
    )

    c2.metric(
        "Actual dropouts identified",
        int(selected["True Positives"])
    )

    c3.metric(
        "Missed dropouts",
        int(selected["Missed Dropouts"])
    )

    c4.metric(
        "False alarms",
        int(selected["False Positives"])
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Precision",
        percent(selected["Precision"])
    )

    c2.metric(
        "Recall",
        percent(selected["Recall"])
    )

    c3.metric(
        "Binary F1",
        percent(selected["F1"])
    )

    st.subheader("Precision and Recall Trade-off")

    st.line_chart(
        model_data.set_index("Threshold")[
            ["Precision", "Recall", "F1"]
        ]
    )

    st.subheader("Intervention Workload")

    st.line_chart(
        model_data.set_index("Threshold")[
            [
                "Students Flagged",
                "True Positives",
                "False Positives"
            ]
        ]
    )

    st.warning(
        "This simulator uses retrospective test "
        "results. Thresholds are exploratory, "
        "not validated operational recommendations."
    )

# =====================================================
# LIMITATIONS AND ETHICS
# =====================================================

elif page == "Limitations and Ethics":

    st.title("Limitations and Responsible Use")

    st.markdown("""
    ### Intended use

    Research and demonstration of how machine
    learning could support academic intervention.

    ### Prohibited use

    Predictions should not be used to deny
    admission, scholarships, financial assistance,
    or educational opportunities.

    ### Dataset limitations

    - Limited institutional and geographic scope.
    - Unequal representation of outcome classes.
    - Small demographic subgroup samples.
    - Potential feature timing and leakage concerns.
    - Integer-coded categorical variables may
      introduce artificial numerical relationships.

    ### Model limitations

    - Retrospective evaluation.
    - No independent external validation.
    - Predicted probabilities are not calibrated.
    - Test-set threshold exploration may produce
      optimistic estimates.
    - Performance differences do not establish
      model fairness.

    ### Future work

    - Independent temporal validation.
    - Probability calibration.
    - Improved categorical feature encoding.
    - Subgroup uncertainty estimation.
    - Validation-based intervention thresholds.
    - Collaboration with academic advisors.

    ### Human oversight

    The model should help identify opportunities
    to provide support. It should never replace
    professional judgment or student autonomy.
    """)

st.divider()

st.caption(
    "Student Dropout Risk | "
    "Learning for Breaking Barriers | "
    "Academic Research Demonstration"
)
