import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)


# ============================================================
# APPLICATION SETTINGS
# ============================================================

st.set_page_config(
    page_title="Shopper Intelligence Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            font-size: 16px;
            color: #6b7280;
            margin-bottom: 25px;
        }

        .section-title {
            font-size: 24px;
            font-weight: 650;
            margin-top: 15px;
            margin-bottom: 10px;
        }

        .model-box {
            padding: 18px;
            border-radius: 12px;
            border: 1px solid #d9dee7;
            background-color: #f8fafc;
            margin-bottom: 15px;
        }

        .metric-label {
            font-size: 14px;
            color: #64748b;
        }

        .metric-value {
            font-size: 26px;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            border: 1px solid #e2e8f0;
            padding: 14px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

FEATURE_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend"
]

CLASSIFIERS = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib"
}

MODEL_DIRECTORY = Path(__file__).resolve().parent / "model"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def get_model(model_name):
    """Load and cache the selected trained model."""
    model_file = MODEL_DIRECTORY / CLASSIFIERS[model_name]

    if not model_file.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_file.name}"
        )

    return joblib.load(model_file)


def validate_dataset(dataframe):
    """Check whether the uploaded CSV contains the required structure."""

    if dataframe.empty:
        return False, "The uploaded CSV does not contain any records."

    if "Revenue" not in dataframe.columns:
        return False, "The target column 'Revenue' is missing."

    absent_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataframe.columns
    ]

    if absent_columns:
        return False, (
            "The following required features are missing: "
            + ", ".join(absent_columns)
        )

    if dataframe["Revenue"].isna().any():
        return False, "The Revenue column contains missing values."

    valid_targets = {True, False, 0, 1}

    if not set(dataframe["Revenue"].unique()).issubset(valid_targets):
        return False, (
            "Revenue must contain only Boolean values "
            "(True/False) or binary values (0/1)."
        )

    return True, "Dataset validation successful."


def calculate_metrics(actual, predicted, probability):
    """Generate all evaluation metrics."""

    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, probability),
        "Precision": precision_score(
            actual,
            predicted,
            zero_division=0
        ),
        "Recall": recall_score(
            actual,
            predicted,
            zero_division=0
        ),
        "F1 Score": f1_score(
            actual,
            predicted,
            zero_division=0
        ),
        "MCC": matthews_corrcoef(
            actual,
            predicted
        )
    }


def draw_confusion_matrix(actual, predicted, model_name):
    """Display the confusion matrix."""

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[False, True]
    )

    fig, axis = plt.subplots(figsize=(6, 4))

    image = axis.imshow(matrix)

    axis.set_title(
        f"Confusion Matrix — {model_name}",
        fontsize=13,
        pad=12
    )

    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    axis.set_xticklabels(
        ["No Purchase", "Purchase"]
    )

    axis.set_yticklabels(
        ["No Purchase", "Purchase"]
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold"
            )

    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛍️ Shopper Intelligence Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        Machine learning based analysis of online shopping sessions
        and purchase behaviour.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Analysis Settings")

    chosen_classifier = st.selectbox(
        "Select a machine learning model",
        options=list(CLASSIFIERS.keys())
    )

    st.divider()

    st.markdown("### Project Information")

    st.write(
        "This application evaluates five classification algorithms "
        "using the trained pipelines stored in the project repository."
    )

    st.caption(
        "Target variable: Revenue"
    )

    st.caption(
        "Classes: No Purchase / Purchase"
    )


# ============================================================
# MODEL LOADING
# ============================================================

try:
    trained_model = get_model(chosen_classifier)

    st.success(
        f"Model ready: {chosen_classifier}"
    )

except Exception as model_error:

    st.error(
        f"Could not load the selected model: {model_error}"
    )

    st.stop()


# ============================================================
# DATA UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">📂 Input Data</div>',
    unsafe_allow_html=True
)

uploaded_csv = st.file_uploader(
    "Choose a CSV file containing the test observations",
    type=["csv"],
    help="Upload test_data.csv or another CSV with the required features."
)


if uploaded_csv is None:

    st.info(
        "Upload a test CSV file to start the model evaluation."
    )

    st.markdown("### Required input")

    feature_preview = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS
        }
    )

    st.dataframe(
        feature_preview,
        use_container_width="stretch",
        hide_index=True
    )

    st.stop()


# ============================================================
# READ AND VALIDATE DATA
# ============================================================

try:

    uploaded_data = pd.read_csv(uploaded_csv)

except Exception as csv_error:

    st.error(
        f"Unable to read the uploaded CSV: {csv_error}"
    )

    st.stop()


is_valid, validation_message = validate_dataset(
    uploaded_data
)

if not is_valid:

    st.error(validation_message)
    st.stop()

st.success(validation_message)


# ============================================================
# PREPARE DATA
# ============================================================

input_data = uploaded_data[FEATURE_COLUMNS]
actual_values = uploaded_data["Revenue"]


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

try:

    predicted_values = trained_model.predict(
        input_data
    )

    purchase_probability = trained_model.predict_proba(
        input_data
    )[:, 1]

except Exception as prediction_error:

    st.error(
        f"Prediction could not be completed: {prediction_error}"
    )

    st.stop()


# ============================================================
# METRIC CALCULATION
# ============================================================

scores = calculate_metrics(
    actual_values,
    predicted_values,
    purchase_probability
)


# ============================================================
# SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">📊 Model Evaluation</div>',
    unsafe_allow_html=True
)

st.caption(
    f"Evaluation performed using {len(uploaded_data):,} records "
    f"with the {chosen_classifier} classifier."
)


metric_row_1 = st.columns(3)
metric_row_2 = st.columns(3)

metric_row_1[0].metric(
    "Accuracy",
    f"{scores['Accuracy']:.2%}"
)

metric_row_1[1].metric(
    "AUC",
    f"{scores['AUC']:.4f}"
)

metric_row_1[2].metric(
    "Precision",
    f"{scores['Precision']:.2%}"
)

metric_row_2[0].metric(
    "Recall",
    f"{scores['Recall']:.2%}"
)

metric_row_2[1].metric(
    "F1 Score",
    f"{scores['F1 Score']:.4f}"
)

metric_row_2[2].metric(
    "MCC",
    f"{scores['MCC']:.4f}"
)


# ============================================================
# TABS
# ============================================================

performance_tab, matrix_tab, report_tab, predictions_tab = st.tabs(
    [
        "📈 Performance",
        "🔲 Confusion Matrix",
        "📋 Classification Report",
        "🔍 Predictions"
    ]
)


# ============================================================
# PERFORMANCE TAB
# ============================================================

with performance_tab:

    st.subheader("Evaluation Summary")

    score_table = pd.DataFrame(
        {
            "Metric": list(scores.keys()),
            "Value": [
                round(scores["Accuracy"], 4),
                round(scores["AUC"], 4),
                round(scores["Precision"], 4),
                round(scores["Recall"], 4),
                round(scores["F1 Score"], 4),
                round(scores["MCC"], 4)
            ]
        }
    )

    st.dataframe(
        score_table,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "AUC measures ranking ability, while F1 balances precision "
        "and recall. MCC provides an additional balanced measure "
        "for classification performance."
    )


# ============================================================
# CONFUSION MATRIX TAB
# ============================================================

with matrix_tab:

    st.subheader("Purchase Classification Matrix")

    draw_confusion_matrix(
        actual_values,
        predicted_values,
        chosen_classifier
    )


# ============================================================
# CLASSIFICATION REPORT TAB
# ============================================================

with report_tab:

    st.subheader("Detailed Classification Report")

    detailed_report = classification_report(
        actual_values,
        predicted_values,
        labels=[False, True],
        target_names=[
            "No Purchase",
            "Purchase"
        ],
        output_dict=True,
        zero_division=0
    )

    report_table = (
        pd.DataFrame(detailed_report)
        .transpose()
        .round(4)
    )

    st.dataframe(
        report_table,
        use_container_width=True
    )


# ============================================================
# PREDICTION TAB
# ============================================================

with predictions_tab:

    st.subheader("Prediction Results")

    prediction_output = uploaded_data.copy()

    prediction_output["Actual Revenue"] = actual_values
    prediction_output["Predicted Revenue"] = predicted_values
    prediction_output["Purchase Probability"] = (
        purchase_probability.round(4)
    )

    st.dataframe(
        prediction_output.head(20),
        use_container_width=True
    )

    csv_output = prediction_output.to_csv(
        index=False
    )

    st.download_button(
        label="⬇️ Download Complete Prediction Results",
        data=csv_output,
        file_name="shopper_prediction_results.csv",
        mime="text/csv"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Online Shopper Purchase Prediction | "
    "Machine Learning Classification Project"
)
