import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_absolute_error

# Title
st.title("💼 Salary Prediction using Multilinear Regression")

# File uploader
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Features & Target
    X = df[['age', 'experience']]
    y = df['salary']

    # Train-test split
    test_size = st.slider("Select Test Size (%)", 10, 50, 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    st.subheader("📈 Model Performance")
    st.write("MAE :",mean_absolute_error(y_test, y_pred))


    # Coefficients
    st.subheader("Model Coefficients")
    st.write("Intercept:", model.intercept_)
    st.write("Age Coefficient:", model.coef_[0])
    st.write("Experience Coefficient:", model.coef_[1])

    # User input prediction
    st.subheader("Predict Salary")

    age_input = st.number_input("Enter Age", min_value=18, max_value=65, value=25)
    exp_input = st.number_input("Enter Experience", min_value=0, max_value=40, value=2)

    if st.button("Predict"):
        prediction = model.predict([[age_input, exp_input]])
        st.success(f"💰 Predicted Salary: ₹{int(prediction[0])}")

else:
    st.info("Please upload your dataset to proceed.")