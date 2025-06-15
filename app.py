import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt
import requests
from streamlit_lottie import st_lottie
import warnings

warnings.filterwarnings("ignore")
pio.templates.default = "plotly_white"

# ─────────────────────────────────────
# 🌐 Page Setup
# ─────────────────────────────────────
st.set_page_config(page_title="Churn Dashboard", page_icon="📊", layout="wide")

# ─────────────────────────────────────
# 🔰 Load Lottie Animation
# ─────────────────────────────────────
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        return None

lottie_icon = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_jcikwtux.json")

st.sidebar.title("📉 Churn Risk Analyzer")
with st.sidebar:
    if lottie_icon:
        st_lottie(lottie_icon, height=160)
    else:
        st.warning("⚠️ Unable to load animation.")

# ─────────────────────────────────────
# 📍 Sidebar Navigation
# ─────────────────────────────────────
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Choose a section", [
    "📖 Introduction", 
    "📈 EDA Dashboard", 
    "🔮 Predict Churn", 
    "📘 Conclusion"
])

# ─────────────────────────────────────
# 📖 Introduction Page
# ─────────────────────────────────────
if page == "📖 Introduction":
    st.title("📖 Project Introduction")
    st.markdown("""
    This project is a part of my university data science coursework. The main objective is to apply machine learning to predict customer churn in the banking sector — that is, identifying customers who are likely to leave the bank.

    For this purpose, I selected the **Customer Churn Modelling dataset from Kaggle**, which contains **10,000 records** of bank customers. The dataset includes features such as Credit Score, Age, Geography, Gender, Balance, Tenure, IsActiveMember, and more. The target variable is `Exited`, indicating whether a customer churned (1) or stayed (0).

    I began by performing **data cleaning**, which involved:
    - Removing irrelevant columns (`RowNumber`, `Surname`, `CustomerId`)
    - Encoding categorical features
    - Scaling numerical features using `StandardScaler`

    After preprocessing, I trained several models and selected the **Random Forest Classifier** based on its accuracy and performance. I then built a complete **Streamlit dashboard** to explore the data and allow users to make churn predictions in real-time.
    """)

# ─────────────────────────────────────
# 📈 EDA Dashboard
# ─────────────────────────────────────
elif page == "📈 EDA Dashboard":
    st.markdown("""
    <h1 style='text-align: center; color: #4B8BBE;'>🏦 Bank Customer Churn Prediction Dashboard</h1>
    <p style='text-align: center; color: gray;'>Explore churn trends and customer behavior through visualizations.</p>
    """, unsafe_allow_html=True)

    # Load data and model
    df = pd.read_csv("clean_churn_data.csv")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📍 Categories", "📊 Numerics", "🔗 Correlation"])

    with tab1:
        st.subheader("🧾 Dataset Preview")
        st.dataframe(df.head())

        st.subheader("📌 Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("🎯 Churn Distribution")
        churn_counts = df["Exited"].value_counts().rename({0: "Stayed", 1: "Churned"})
        fig = px.bar(
            x=churn_counts.index,
            y=churn_counts.values,
            text_auto=True,
            labels={"x": "Status", "y": "Count"},
            color=churn_counts.index,
            color_discrete_sequence=["#00cc96", "#EF553B"],
            title="Churn vs Non-Churn"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("📍 Categorical Features vs Churn")
        for feature in ["Geography", "Gender", "HasCrCard", "IsActiveMember"]:
            ct = pd.crosstab(df[feature], df["Exited"], normalize="index") * 100
            ct = ct.rename(columns={0: "Stayed", 1: "Churned"}).reset_index()
            ct = ct.melt(id_vars=feature, value_name="value", var_name="Status")

            fig = px.bar(
                ct, x=feature, y="value", color="Status", barmode="group",
                text_auto=".2f", labels={"value": "Percentage"},
                title=f"{feature} vs Churn (%)",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("📊 Numerical Feature Distributions")
        for col in ["CreditScore", "Age", "Balance", "EstimatedSalary"]:
            fig = px.histogram(
                df, x=col, color="Exited", barmode="overlay", histnorm="percent",
                marginal="box", nbins=40,
                title=f"{col} Distribution by Churn",
                labels={"Exited": "Churn"},
                color_discrete_sequence=["#00cc96", "#EF553B"]
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("🔗 Correlation Heatmap")
        df_corr = df.copy()
        df_corr["Gender"] = df_corr["Gender"].map({"Male": 1, "Female": 0})
        df_corr["Geography"] = df_corr["Geography"].map({"France": 0, "Germany": 1, "Spain": 2})
        corr = df_corr.select_dtypes(include=["int64", "float64"]).corr()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="RdBu", center=0, fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)

# ─────────────────────────────────────
# 🔮 Prediction Page
# ─────────────────────────────────────
elif page == "🔮 Predict Churn":
    st.title("🔮 Predict Bank Customer Churn")

    # Load model and scaler
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        credit_score     = col1.number_input("Credit Score", 300, 900, 650)
        geography        = col2.selectbox("Geography", ["France", "Germany", "Spain"])
        gender           = col1.selectbox("Gender", ["Male", "Female"])
        age              = col2.slider("Age", 18, 100, 35)
        tenure           = col1.slider("Tenure (Years)", 0, 10, 3)
        balance          = col2.number_input("Balance", 0.0, 300000.0, 50000.0)
        num_products     = col1.selectbox("Number of Products", [1, 2, 3, 4])
        has_cr_card      = col2.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = col1.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = col2.number_input("Estimated Salary", 0.0, 300000.0, 60000.0)

        submitted = st.form_submit_button("🚀 Predict")

    if submitted:
        gender = 1 if gender == "Male" else 0
        geo_map = {"France": 0, "Germany": 1, "Spain": 2}
        geography = geo_map[geography]
        has_cr_card = 1 if has_cr_card == "Yes" else 0
        is_active_member = 1 if is_active_member == "Yes" else 0

        input_data = np.array([[credit_score, geography, gender, age, tenure,
                                balance, num_products, has_cr_card,
                                is_active_member, estimated_salary]])

        columns = ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure',
                   'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']
        input_df = pd.DataFrame(input_data, columns=columns)
        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        prob = model.predict_proba(scaled_input)[0][1]

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ This customer is likely to **CHURN**. (Probability: {prob:.2f})")
        else:
            st.success(f"✅ This customer is likely to **STAY**. (Churn Probability: {prob:.2f})")

# ─────────────────────────────────────
# 📘 Conclusion Page
# ─────────────────────────────────────
elif page == "📘 Conclusion":
    st.title("📘 Project Conclusion")
    st.markdown("""
    Through this project, I successfully built a machine learning model to predict customer churn using real-world banking data. The process involved data preprocessing, feature encoding, model training, and evaluation.

    One of the challenges was handling categorical variables like `Geography` and ensuring proper scaling for numerical data. I experimented with different models, and the **Random Forest Classifier** gave the best results in terms of accuracy and generalization.

    The final model was deployed using a **Streamlit app** that combines:
    - An interactive EDA dashboard using Plotly
    - A real-time prediction interface

    This project helped me understand how to turn raw data into a usable tool that can support business decisions. It was a great learning experience in practical machine learning, model deployment, and building user-friendly data applications.
    """)

# ─────────────────────────────────────
# 📌 Footer
# ─────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Developed by <strong>Amos Shehzad</strong> | Data Science - PUCIT</p>",
    unsafe_allow_html=True
)
