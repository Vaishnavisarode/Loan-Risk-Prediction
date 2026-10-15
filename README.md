# LoanGuard — Explainable Loan Risk Prediction

LoanGuard is an end-to-end machine learning project that estimates loan default probability and explains the factors influencing individual predictions using SHAP.

## Project Overview

The goal is to demonstrate a complete ML workflow, from exploratory data analysis and feature engineering to model evaluation, explainability, and a Streamlit application.

## Features

- Loan default probability prediction
- Logistic Regression classification pipeline
- Loan-to-Income Ratio feature engineering
- Model evaluation using precision, recall, F1-score, ROC-AUC, and PR-AUC
- Configurable prediction threshold
- SHAP-based global and individual explanations
- Interactive Streamlit application

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, SHAP, Matplotlib, Seaborn, XGBoost, Streamlit, and Jupyter Notebook.

## Project Structure

- `app.py` — Streamlit application
- `models/` — saved model and configuration artifacts
- `notebooks/` — EDA, feature engineering, training, evaluation, explainability, and testing
- `requirements.txt` — project dependencies

The source dataset is not included in this repository. Follow the dataset provider's instructions and license to obtain it.

## Model Evaluation

The baseline experiments compared Logistic Regression, Random Forest, and XGBoost. Logistic Regression achieved the highest baseline PR-AUC among those experiments. Threshold tuning was then used to explore the precision-recall trade-off.

The current application uses a Logistic Regression model with a 0.65 decision threshold. This threshold is an experimental project setting, not a validated lending policy.

## Explainability

SHAP is used to examine which features increase or decrease the model's output for an individual application. These contributions describe model behavior and do not establish causal relationships.

## Run Locally

1. Clone the repository.
2. Create and activate a Python virtual environment.
3. Install dependencies with `pip install -r requirements.txt`.
4. Obtain the dataset separately if you want to rerun the notebooks.
5. Run the application:

   `streamlit run app.py`

The saved model and configuration artifacts must be present in the `models/` directory.

## Limitations

This is an educational portfolio project, not a validated credit-scoring system. Predictions may be inaccurate or biased and must not be used alone to make real lending decisions.

## Author

Add your name and links to your GitHub profile and LinkedIn here.
