import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/churn_model.pkl")
columns = joblib.load("models/columns.pkl")

# -----------------------------
# Title
# -----------------------------
st.title("📊 Customer Churn Prediction System")
st.markdown("### Predict whether a customer is likely to leave the company.")

st.divider()

# -----------------------------
# Customer Details
# -----------------------------
st.header("👤 Customer Details")

gender = st.selectbox("Gender", ["Male", "Female"])

senior = st.selectbox(
    "Senior Citizen",
    [0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No"
)

partner = st.selectbox("Partner", ["Yes", "No"])

dependents = st.selectbox("Dependents", ["Yes", "No"])

tenure = st.slider(
    "Tenure (Months)",
    min_value=0,
    max_value=72,
    value=12
)

monthly = st.number_input(
    "Monthly Charges",
    min_value=0.0,
    max_value=200.0,
    value=70.0
)

total = st.number_input(
    "Total Charges",
    min_value=0.0,
    max_value=10000.0,
    value=1000.0
)

st.divider()

# -----------------------------
# Prediction Button
# -----------------------------
if st.button("🔍 Predict Customer Churn"):

    # Create input dataframe
    input_data = pd.DataFrame(0, index=[0], columns=columns)

    # Numeric Features
    if "SeniorCitizen" in columns:
        input_data["SeniorCitizen"] = senior

    if "tenure" in columns:
        input_data["tenure"] = tenure

    if "MonthlyCharges" in columns:
        input_data["MonthlyCharges"] = monthly

    if "TotalCharges" in columns:
        input_data["TotalCharges"] = total

    # Categorical Features
    if "gender_Male" in columns and gender == "Male":
        input_data["gender_Male"] = 1

    if "Partner_Yes" in columns and partner == "Yes":
        input_data["Partner_Yes"] = 1

    if "Dependents_Yes" in columns and dependents == "Yes":
        input_data["Dependents_Yes"] = 1

    # Prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    stay_probability = probability[0][0] * 100
    churn_probability = probability[0][1] * 100

    st.divider()

    # -----------------------------
    # Prediction Result
    # -----------------------------
    st.header("📈 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ Customer is likely to Churn")
    else:
        st.success("✅ Customer is NOT likely to Churn")

    # -----------------------------
    # Probabilities
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.metric("🟢 Stay Probability", f"{stay_probability:.2f}%")

    with col2:
        st.metric("🔴 Churn Probability", f"{churn_probability:.2f}%")

    # -----------------------------
    # Risk Level
    # -----------------------------
    st.subheader("🚦 Risk Level")

    if churn_probability < 30:
        st.success("🟢 LOW RISK")

    elif churn_probability < 70:
        st.warning("🟡 MEDIUM RISK")

    else:
        st.error("🔴 HIGH RISK")

    # -----------------------------
    # Customer Summary
    # -----------------------------
    st.subheader("📋 Customer Summary")

    st.write(f"*Gender:* {gender}")
    st.write(f"*Senior Citizen:* {'Yes' if senior == 1 else 'No'}")
    st.write(f"*Partner:* {partner}")
    st.write(f"*Dependents:* {dependents}")
    st.write(f"*Tenure:* {tenure} Months")
    st.write(f"*Monthly Charges:* ${monthly:.2f}")
    st.write(f"*Total Charges:* ${total:.2f}")

    # -----------------------------
    # Business Recommendation
    # -----------------------------
    st.subheader("💡 Business Recommendation")

    if churn_probability < 30:
        st.info("""
### Recommended Actions

✅ Customer is loyal.

- Offer loyalty rewards.
- Send thank-you emails.
- Maintain excellent customer service.
- Recommend premium plans.
        """)

    elif churn_probability < 70:
        st.warning("""
### Recommended Actions

Customer is at medium risk.

- Offer discount coupons.
- Contact the customer.
- Recommend a yearly subscription.
- Provide better support.
        """)

    else:
        st.error("""
### Recommended Actions

Customer is at HIGH RISK.

- Contact the customer immediately.
- Offer a special retention discount.
- Assign a dedicated support executive.
- Provide personalized offers.
- Follow up within 24 hours.
        """)

    st.success("🎉 Prediction Completed Successfully!")
    st.divider()

st.header("📊 Dashboard")

# Load Dataset
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Churn Distribution
st.subheader("Customer Churn Distribution")
st.bar_chart(df["Churn"].value_counts())

# Contract Type
st.subheader("Contract Type Distribution")
st.bar_chart(df["Contract"].value_counts())

# Internet Service
st.subheader("Internet Service Distribution")
st.bar_chart(df["InternetService"].value_counts())
st.divider()

st.header("ℹ️ About This Project")

st.write("""
This Customer Churn Prediction System uses Machine Learning to predict
whether a customer is likely to leave a telecom company.

### Technologies Used
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

### Machine Learning Algorithm
- Logistic Regression

### Model Accuracy
- 78.54%

### Project Objective
To help telecom companies identify customers who are likely to churn so
they can take action to retain them.
""")