import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dashboard", page_icon="📊")

st.title("📊 Customer Churn Dashboard")

df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")

# -----------------------
# Dataset Information
# -----------------------

st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", len(df))
col2.metric("Columns", len(df.columns))
col3.metric("Churn Rate", f"{(df['Churn'].value_counts(normalize=True)['Yes']*100):.1f}%")

# -----------------------
# Pie Chart
# -----------------------

st.subheader("Customer Churn Distribution")

fig, ax = plt.subplots()

df["Churn"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax
)

ax.set_ylabel("")

st.pyplot(fig)

# -----------------------
# Contract Type
# -----------------------

st.subheader("Contract Types")

fig2, ax2 = plt.subplots()

df["Contract"].value_counts().plot(
    kind="bar",
    ax=ax2
)

ax2.set_xlabel("Contract Type")
ax2.set_ylabel("Customers")

st.pyplot(fig2)

# -----------------------
# Internet Service
# -----------------------

st.subheader("Internet Service")

fig3, ax3 = plt.subplots()

df["InternetService"].value_counts().plot(
    kind="bar",
    ax=ax3
)

st.pyplot(fig3)