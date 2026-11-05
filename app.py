
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

# Page configuration
st.set_page_config(
    page_title="Online Shoppers Purchase Prediction",
    page_icon="🛒",
    layout="wide"
)

st.title("Online Shoppers Purchase Prediction")
st.write(
    "An interactive application to evaluate five machine learning "
    "classification models on online shopping data."
)

# Expected input features
expected_features = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend"
]

# Available trained models
model_files = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "SVM": "svm.joblib"
}

# Model selection
st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Choose a classifier",
    list(model_files.keys())
)

model_path = (
    Path(__file__).parent
    / "model"
    / model_files[selected_model]
)

# Load saved pipeline
try:
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Unable to load selected model: {e}")
    st.stop()

st.success(f"{selected_model} loaded successfully!")

# Dataset upload
st.header("Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("Uploaded dataset is empty.")
            st.stop()

        if "Revenue" not in df.columns:
            st.error("CSV must contain the Revenue target column.")
            st.stop()

        missing_features = [
            col for col in expected_features
            if col not in df.columns
        ]

        if missing_features:
            st.error(f"Missing features: {missing_features}")
            st.stop()

        X_test = df[expected_features]
        y_test = df["Revenue"]

        # Validate target values
        if y_test.isna().any():
            st.error("Revenue contains missing values.")
            st.stop()

        if not set(y_test.unique()).issubset({True, False, 0, 1}):
            st.error("Revenue must contain only True/False or 0/1.")
            st.stop()

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        precision = precision_score(
            y_test, y_pred, zero_division=0
        )
        recall = recall_score(
            y_test, y_pred, zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred, zero_division=0
        )
        mcc = matthews_corrcoef(y_test, y_pred)

        st.header("Model Performance")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.2%}")
        col2.metric("AUC", f"{auc:.4f}")
        col3.metric("Precision", f"{precision:.2%}")
        col4.metric("Recall", f"{recall:.2%}")
        col5.metric("F1 Score", f"{f1:.4f}")
        col6.metric("MCC", f"{mcc:.4f}")

        # Confusion matrix
        st.header("Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=[False, True]
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.imshow(cm, cmap="Blues")

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Purchase", "Purchase"])
        ax.set_yticklabels(["No Purchase", "Purchase"])

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(selected_model)

        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, cm[i, j],
                    ha="center",
                    va="center"
                )

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Classification report
        st.header("Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=[False, True],
            target_names=["No Purchase", "Purchase"],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()

        st.dataframe(
            report_df.round(4),
            use_container_width=True
        )

        # Prediction results
        st.header("Prediction Results")

        results_df = df.copy()
        results_df["Actual Revenue"] = y_test
        results_df["Predicted Revenue"] = y_pred
        results_df["Purchase Probability"] = y_prob

        st.dataframe(results_df.head(10))

        csv_data = results_df.to_csv(index=False)

        st.download_button(
            label="Download Prediction Results",
            data=csv_data,
            file_name="prediction_results.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Evaluation failed: {e}")

else:
    st.info("Upload test_data.csv to evaluate the selected model.")
