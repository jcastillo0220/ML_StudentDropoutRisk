# Model Card: Student Dropout Risk

## Model Overview

Project: Learning for Breaking Barriers

Task: Multiclass classification of student
outcomes into Dropout, Enrolled, and Graduate.

Algorithms:
- Logistic Regression
- Random Forest
- XGBoost

Each algorithm was evaluated using baseline
and optimized configurations.

## Intended Use

Academic research and demonstration of how
machine learning could support early student
intervention and reduce educational barriers.

Predictions should assist academic advisors
rather than replace professional judgment.

## Inappropriate Use

The models must not be used to deny admission,
scholarships, financial assistance, or access
to educational opportunities.

The system is not validated for operational
student-level decision-making.

## Training Data

Source: UCI Predict Students' Dropout and
Academic Success, dataset ID 697.

- Records: 4,424
- Input features: 36
- Missing values: 0
- Duplicate records: 0
- Graduate: 2,209
- Dropout: 1,421
- Enrolled: 794

## Evaluation Method

- Stratified 80/20 train-test split
- Training samples: 3,539
- Testing samples: 885
- Random seed: 42
- Five-fold stratified cross-validation
- Hyperparameter selection using macro F1

## Performance

| Model | Accuracy | Macro F1 | Dropout Recall | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline Logistic Regression | 76.84% | 68.26% | 76.76% | 87.78% |
| Baseline Random Forest | 76.84% | 69.02% | 74.30% | 88.69% |
| Baseline XGBoost | 76.16% | 69.64% | 74.30% | 88.45% |
| Tuned Logistic Regression | 72.99% | 69.36% | 68.31% | 87.68% |
| Tuned Random Forest | 74.69% | 70.34% | 69.72% | 88.55% |
| Tuned XGBoost | 76.16% | 69.39% | 72.54% | 88.79% |

Tuned Random Forest achieved the highest
cross-validation macro F1 of 72.14%.

Baseline Logistic Regression achieved the
highest full-information dropout recall.

## Early Prediction

Three feature-availability scenarios were
evaluated:

- Enrollment: 22 features
- First semester: 28 features
- Full dataset: 36 features

First-semester XGBoost achieved 71.41%
accuracy and 67.61% dropout recall.

These scenarios approximate feature
availability and require timing verification.

## Subgroup Performance

Performance was evaluated across gender,
scholarship status, debt status, tuition
payment status, international status,
and educational special needs.

Tuned Random Forest dropout recall:

- Gender group 0: 69.78%
- Gender group 1: 69.66%
- Non-debtors: 65.16%
- Debtors: 85.71%
- Tuition not up to date: 98.92%
- Tuition up to date: 55.50%

Some groups contain limited sample sizes.
Differences do not establish causation or
prove that the models are fair.

## Risks and Limitations

- Limited institutional generalizability.
- Retrospective experimental design.
- Unequal class representation.
- Small demographic subgroup samples.
- Potential temporal feature leakage.
- Integer-coded categorical features.
- No external validation.
- No probability calibration.
- Hyperparameters reused across scenarios.
- Repeated inspection of the test results.
- Exploratory test-set threshold analysis.

## Ethical Considerations

False negatives may prevent students from
receiving timely support. False positives
may create unnecessary intervention workload
or stigmatize students.

Interventions should be supportive,
voluntary, and reviewed by qualified staff.

## Privacy

The dashboard displays aggregate results.
Individual prediction records, raw data,
and serialized models are excluded from
version control.

## Future Work

- Independent temporal validation
- Improved categorical encoding
- Probability calibration
- Validation-based threshold selection
- Subgroup uncertainty estimation
- Collaboration with academic advisors

## Reproducibility

Run:

    python run_pipeline.py

Launch the research dashboard:

    python -m streamlit run app.py --server.address 127.0.0.1
