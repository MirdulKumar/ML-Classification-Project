# Online Shoppers Purchasing Intention – Machine Learning Classification

## 1. Project Overview

This project aims to predict whether an online visitor will make a purchase based on their browsing behaviour and session information.

The project implements five supervised machine learning classification algorithms, evaluates their performance using multiple classification metrics, and provides an interactive web application developed using Streamlit.

The application allows users to select a trained model, upload a test dataset, view model performance and download prediction results.

## 2. Problem Statement

Understanding customer purchasing behaviour is important for e-commerce businesses. Identifying visitors who are likely to make a purchase can help businesses improve marketing strategies, customer engagement and conversion rates.

The objective of this project is to develop and compare multiple classification models to predict the target variable `Revenue`, which indicates whether a visitor completed a purchase.

## 3. Dataset Description

**Dataset:** Online Shoppers Purchasing Intention

**Source:** UCI Machine Learning Repository

**Dataset link:** https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset

The dataset contains information about online shopping sessions.

| Description | Value |
|---|---|
| Total records | 12,330 |
| Total features | 17 |
| Target variable | Revenue |
| Problem type | Binary Classification |
| Target classes | True / False |

The target variable represents:
- `True`: Visitor made a purchase.
- `False`: Visitor did not make a purchase.

The dataset contains numerical and categorical features describing visitor activities, including administrative pages, informational pages, product-related activities, bounce rates, exit rates, page values, visitor type and weekend visits.

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Seaborn (EDA)
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## 5. Project Workflow

The project follows these stages:

1. Data loading and exploration.
2. Data cleaning and statistical analysis.
3. Identification of numerical and categorical features.
4. Train-test data preparation.
5. Feature preprocessing using StandardScaler and OneHotEncoder.
6. Training five classification algorithms.
7. Model evaluation using six performance metrics.
8. Saving trained models using Joblib.
9. Developing an interactive Streamlit application.
10. Deployment using Streamlit Community Cloud.

## 6. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Each model was implemented using a Scikit-learn Pipeline to combine preprocessing and classification.

The preprocessing includes:
- StandardScaler for numerical features.
- OneHotEncoder for categorical features.

The fitted preprocessing and classification pipelines were saved as Joblib files for reuse in the Streamlit application.

## 7. Model Performance Comparison

The following table presents the performance of all five classifiers on the independent test dataset.

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 83.33% | 89.14% | 47.58% | 74.61% | 58.10% | 0.5023 |
| Decision Tree | 83.37% | 87.50% | 47.86% | 81.94% | 60.42% | 0.5375 |
| KNN | 86.86% | 77.50% | 62.83% | 37.17% | 46.71% | 0.4156 |
| Naive Bayes | 74.21% | 77.76% | 33.42% | 67.02% | 44.60% | 0.3326 |
| SVM | 86.21% | 90.07% | 53.96% | 74.87% | 62.72% | 0.5563 |

## 8. Model Observations

**Logistic Regression**

- Achieved 83.33% accuracy and 89.14% AUC.
- Provided a reasonable baseline classification performance.
- Achieved 74.61% recall for the Purchase class.

**Decision Tree**

- Achieved the highest Purchase-class recall of 81.94%.
- Identified the largest proportion of actual purchasing customers.
- Achieved an F1-score of 60.42%.

**K-Nearest Neighbors (KNN)**

- Achieved the highest accuracy of 86.86%.
- Achieved the highest Purchase-class precision of 62.83%.
- However, its recall was relatively low at 37.17%.

**Naive Bayes**

- Achieved 74.21% accuracy.
- Showed comparatively weaker overall performance among the five classifiers.
- Achieved 67.02% Purchase-class recall.

**Support Vector Machine (SVM)**

- Achieved the highest AUC of 90.07%.
- Achieved the highest F1-score of 62.72%.
- Achieved the highest MCC of 0.5563.
- Demonstrated the strongest overall balance between precision and recall.

## 9. Overall Model Selection

Based on the independent test dataset, **Support Vector Machine (SVM)** was selected as the best overall classifier.

It achieved the highest AUC, F1-score and Matthews Correlation Coefficient (MCC), indicating a strong balance in classification performance.

However, Decision Tree achieved the highest recall of 81.94%, making it a suitable alternative when identifying the maximum number of potential purchasing customers is the primary objective.

Since the dataset is imbalanced, accuracy alone was not considered sufficient for selecting the final model.

## 10. Streamlit Web Application

An interactive Streamlit application was developed to demonstrate the trained classification models.

Application features:

- Selection of five trained classification models.
- Upload of test data in CSV format.
- Automatic prediction using the selected saved model.
- Display of Accuracy, AUC, Precision, Recall, F1-score and MCC.
- Confusion matrix visualization.
- Detailed classification report.
- Prediction results preview.
- Downloadable prediction results in CSV format.

The application loads previously trained models and does not perform model training during evaluation.

## 11. Project Structure

```text
ML-Classification-Project/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    └── 02_model_training_evaluation.ipynb
```

## 12. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/MirdulKumar/ML-Classification-Project.git
```

### Step 2: Navigate to the project directory

```bash
cd ML-Classification-Project
```

### Step 3: Create a virtual environment

```bash
python -m venv .venv
```

### Step 4: Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\activate
```

### Step 5: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 13. Project Links

- **GitHub Repository:** https://github.com/MirdulKumar/ML-Classification-Project
- **Live Streamlit Application:** To be added after deployment.

## 14. Conclusion

This project demonstrates an end-to-end machine learning classification workflow, from data exploration and preprocessing to model training, evaluation and interactive deployment.

Five classification algorithms were implemented and compared using multiple evaluation metrics. SVM demonstrated the strongest overall performance, while Decision Tree achieved the highest recall for the Purchase class.

The Streamlit application provides an interactive interface to evaluate the trained models and explore their predictions on uploaded test datasets.
