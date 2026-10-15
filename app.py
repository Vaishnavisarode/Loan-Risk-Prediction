import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
from pathlib import Path


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "logistic_baseline.pkl"
CONFIG_PATH = BASE_DIR / "models" / "model_config.json"
SHAP_BACKGROUND_PATH = (
    BASE_DIR / "models" / "shap_background.pkl"
)

SHAP_FEATURE_NAMES_PATH = (
    BASE_DIR / "models" / "shap_feature_names.pkl"
)
# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="LoanGuard",
    page_icon="🛡️",
    layout="wide"
)

# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
}

.subtitle {
    font-size: 18px;
    color: #6b7280;
    margin-top: 0;
}

.section-title {
    font-size: 24px;
    font-weight: 600;
    margin-top: 10px;
}

.metric-card {
    padding: 20px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    text-align: center;
}

.metric-label {
    font-size: 14px;
    color: #6b7280;
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
}

.info-card {
    padding: 18px;
    border-radius: 12px;
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    margin-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():

    model = joblib.load(MODEL_PATH)

    return model


@st.cache_data
def load_config():

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    return config

@st.cache_resource
def load_shap_explainer():

    background = joblib.load(
        SHAP_BACKGROUND_PATH
    )

    feature_names = joblib.load(
        SHAP_FEATURE_NAMES_PATH
    )

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    explainer = shap.LinearExplainer(
        classifier,
        background
    )

    return explainer, feature_names


model = load_model()
config = load_config()

THRESHOLD = config["threshold"]
explainer, feature_names = load_shap_explainer()

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ LoanGuard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Explainable Loan Default Risk Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.write(
    """
    Estimate loan default probability using machine learning
    and understand the factors influencing each prediction.
    """
)

st.divider()

# ============================================================
# LOAN APPLICATION FORM
# ============================================================

st.header("📋 Loan Application")

col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# Personal Information
# ------------------------------------------------------------

with col1:

    st.subheader("Personal Information")

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=69,
        value=30
    )

    income = st.number_input(
        "Annual Income",
        min_value=15000,
        max_value=149999,
        value=50000
    )

    education = st.selectbox(
        "Education",
        [
            "High School",
            "Bachelor's",
            "Master's",
            "PhD"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )


# ------------------------------------------------------------
# Employment Information
# ------------------------------------------------------------

with col2:

    st.subheader("Employment & Credit")

    months_employed = st.number_input(
        "Months Employed",
        min_value=0,
        max_value=119,
        value=36
    )

    employment_type = st.selectbox(
        "Employment Type",
        [
            "Full-time",
            "Part-time",
            "Self-employed",
            "Unemployed"
        ]
    )

    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=849,
        value=650
    )

    num_credit_lines = st.number_input(
        "Number of Credit Lines",
        min_value=1,
        max_value=4,
        value=2
    )


# ------------------------------------------------------------
# Loan Information
# ------------------------------------------------------------

with col3:

    st.subheader("Loan Information")

    loan_amount = st.number_input(
        "Loan Amount",
        min_value=5000,
        max_value=249999,
        value=50000
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=2.0,
        max_value=25.0,
        value=10.0
    )

    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=12,
        max_value=60,
        value=36
    )

    dti_ratio = st.number_input(
        "DTI Ratio",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.01
    )


st.divider()


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

st.header("🏠 Additional Information")

col1, col2, col3, col4 = st.columns(4)

with col1:

    has_mortgage = st.selectbox(
        "Has Mortgage?",
        ["Yes", "No"]
    )

with col2:

    has_dependents = st.selectbox(
        "Has Dependents?",
        ["Yes", "No"]
    )

with col3:

    loan_purpose = st.selectbox(
        "Loan Purpose",
        [
            "Business",
            "Home",
            "Education",
            "Other",
            "Auto"
        ]
    )

with col4:

    has_cosigner = st.selectbox(
        "Has Co-Signer?",
        ["Yes", "No"]
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Analyze Loan Risk",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

    loan_to_income_ratio = loan_amount / income


    # --------------------------------------------------------
    # Create input dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Age": [age],
        "Income": [income],
        "LoanAmount": [loan_amount],
        "CreditScore": [credit_score],
        "MonthsEmployed": [months_employed],
        "NumCreditLines": [num_credit_lines],
        "InterestRate": [interest_rate],
        "LoanTerm": [loan_term],
        "DTIRatio": [dti_ratio],
        "Education": [education],
        "EmploymentType": [employment_type],
        "MaritalStatus": [marital_status],
        "HasMortgage": [has_mortgage],
        "HasDependents": [has_dependents],
        "LoanPurpose": [loan_purpose],
        "HasCoSigner": [has_cosigner],
        "LoanToIncomeRatio": [loan_to_income_ratio]
    })


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_data
    )[0, 1]

    prediction = int(
        probability >= THRESHOLD
    )


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    # ============================================================
    # RISK ASSESSMENT
    # ============================================================

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Loan Risk Assessment</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2, result_col3 = st.columns(3)


    # ------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------

    with result_col1:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Default Probability
                </div>
                <div class="metric-value">
                    {probability:.1%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ------------------------------------------------------------
    # Risk Decision
    # ------------------------------------------------------------

    with result_col2:

        if prediction == 1:

            risk_text = "Higher Risk"
            risk_icon = "⚠️"

        else:

            risk_text = "Lower Risk"
            risk_icon = "✅"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">
                Model Decision
            </div>
            <div class="metric-value">
                {risk_icon} {risk_text}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    # ------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------

    with result_col3:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Decision Threshold
                </div>
                <div class="metric-value">
                    {THRESHOLD:.0%}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    st.write("")

    st.progress(    
    min(float(probability), 1.0)
)

    st.caption( 
    "The probability is compared with the model's "
    f"{THRESHOLD:.0%} decision threshold."
)
    # ============================================================
    # SHAP EXPLANATION
    # ============================================================

    st.divider()

    st.header(  
        "🧠 Why did the model make this prediction?"
    )

    st.caption(
        "   These factors show which features pushed the model's "
        "predicted default probability higher or lower for this "
        "application."
    )   

    # Transform applicant using the same preprocessing
    input_transformed = model.named_steps[
        "preprocessor"
    ].transform(input_data)

    # Calculate SHAP values
    input_shap = explainer.shap_values(
        input_transformed
    )[0]

    # Create explanation dataframe
    explanation_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": input_shap
    })

    explanation_df["Absolute_Impact"] = (
        explanation_df["SHAP_Value"].abs()
    )

    # Convert technical feature names into readable names
    def clean_feature_name(feature):

        feature = feature.replace("num_", "")
        feature = feature.replace("cat_", "")
        feature = feature.replace("_", " ")

        return feature


    explanation_df["Feature"] = (
        explanation_df["Feature"]
        .apply(clean_feature_name)
    )

    # Sort by strongest impact
    explanation_df = explanation_df.sort_values(
        "Absolute_Impact",
        ascending=False
    )

    # Positive and negative factors
    positive_factors = explanation_df[
        explanation_df["SHAP_Value"] > 0
    ].head(3)

    negative_factors = explanation_df[
        explanation_df["SHAP_Value"] < 0
    ].head(3)


    col1, col2 = st.columns(2)


    with col1:

        st.subheader(
            "🔺 Increasing predicted default risk"
        )

        for _, row in positive_factors.iterrows():

            st.markdown(
                f"🔺 **{row['Feature']}**  \n"
                f"SHAP contribution: `{row['SHAP_Value']:.3f}`"
            )


    with col2:

        st.subheader(
            "🔻 Decreasing predicted default risk"
        )

        for _, row in negative_factors.iterrows():

            st.markdown(
                f"🔻 **{row['Feature']}**  \n"
                f"SHAP contribution: `{row['SHAP_Value']:.3f}`"
            )
    st.divider()

    with st.expander("ℹ️ About LoanGuard"):

        st.write(
            """
            **LoanGuard** is an explainable machine learning
            application for loan default risk prediction.

            **Model:** Logistic Regression

            **Decision threshold:** 65%

            **Feature engineering:** Loan-to-Income Ratio

            **Explainability:** SHAP

            The prediction is a machine learning estimate and
            should not be treated as a guarantee of future
            repayment or default.
            """
        )