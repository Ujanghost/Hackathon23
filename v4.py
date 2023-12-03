import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor

# Landing Page
st.set_option('deprecation.showPyplotGlobalUse', False)



st.image("https://lavasa.christuniversity.in/images/logo.png", caption="", use_column_width=True)
st.title("Welcome to EduInsight")
st.write("EduInsight Empowers Students and Educators with data driven insights to enhance the education journey and academic success!👨‍🎓📊📈")
# st.image("hackathon final/logo.png", caption="Image Caption 2", use_column_width=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "EduQuality", "SocialEdu","ParentEduConnect","EduSip"])

# Load the dataset
data_description = """
school -
sex - student's sex (binary: 'F' - female or 'M' - male)
... (other attributes)
G3 - final grade (numeric: from 0 to 20)
"""

# Home Page
if page == "Home":

    a=1+1
    # st.subheader("Welcome to Student Performance Prediction App")
    # st.write(
    #     "This app helps you explore the data and predict the final grades of students based on various features."
    # )

# Data Exploration Page
elif page == "EduQuality":
    st.subheader("Data Exploration")
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Description:")
        st.text(data_description)
        st.subheader("Sample Data:")
        st.write(data.head())

        st.subheader("Correlation Matrix:")
        correlation_matrix = data.corr()



        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix[['G3']], annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix for Final Grade (G3)')
        plt.show()
        st.subheader("Model Prediction")
    # uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    # if uploaded_file is not None:
        # data = pd.read_csv(uploaded_file)
        target_variable = st.selectbox("Select the target variable", data.columns)
        feature_columns = st.multiselect("Select the input features", data.columns[:-1])
        X = data[feature_columns]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.subheader("Select a model:")
        model_name = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network","Gradient Boosing",])
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor()
        elif model_name == "XGBoost":
            model = XGBRegressor()
        elif model_name == "Neural Network":
            model = MLPRegressor(max_iter=100)
        elif model_name=="Gradient boosting":
            model= GradientBoostingRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.subheader(f"Model Evaluation (Mean Squared Error): {mse:.2f}")
        r_sq= r2_score(y_test, y_pred)
        st.subheader(f"R-squared Error:{r_sq:.2f}")
        st.subheader("Model Predictions vs Actual Values:")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write(results)
        st.subheader("Train vs Test Performance:")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.index, y_test, label="Actual", alpha=0.7)
        plt.scatter(y_test.index, y_pred, label="Predicted", alpha=0.7)
        plt.title("Actual vs Predicted on Test Set")
        plt.xlabel("Data Points")
        plt.ylabel(target_variable)
        plt.legend()
        st.pyplot()
        

# Model Prediction Page
elif page == "SocialEdu":
    st.subheader("Model Prediction")
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        target_variable = st.selectbox("Select the target variable", data.columns)
        feature_columns = st.multiselect("Select the input features", data.columns[:-1])
        X = data[feature_columns]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.subheader("Select a model:")
        model_name = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"])
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor()
        elif model_name == "XGBoost":
            model = XGBRegressor()
        elif model_name == "Neural Network":
            model = MLPRegressor(max_iter=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.subheader(f"Model Evaluation (Mean Squared Error): {mse:.2f}")
        r_sq= r2_score(y_test, y_pred)
        st.subheader(f"R-squared Error:{r_sq:.2f}")
        st.subheader("Model Predictions vs Actual Values:")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write(results)
        st.subheader("Train vs Test Performance:")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.index, y_test, label="Actual", alpha=0.7)
        plt.scatter(y_test.index, y_pred, label="Predicted", alpha=0.7)
        plt.title("Actual vs Predicted on Test Set")
        plt.xlabel("Data Points")
        plt.ylabel(target_variable)
        plt.legend()
        st.pyplot()
elif page=="ParentEduConnect":
    st.subheader("Data Exploration")
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Description:")
        st.text(data_description)
        st.subheader("Sample Data:")
        st.write(data.head())

        st.subheader("Correlation Matrix:")
        correlation_matrix = data.corr()



        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix[['G3']], annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix for Final Grade (G3)')
        plt.show()
        st.subheader("Model Prediction")
# uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
# if uploaded_file is not None:
    # data = pd.read_csv(uploaded_file)
    target_variable = st.selectbox("Select the target variable", data.columns)
    feature_columns = st.multiselect("Select the input features", data.columns[:-1])
    X = data[feature_columns]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.subheader("Select a model:")
    model_name = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network","Gradient Boosing",])
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor()
    elif model_name == "XGBoost":
        model = XGBRegressor()
    elif model_name == "Neural Network":
        model = MLPRegressor(max_iter=100)
    elif model_name=="Gradient boosting":
        model= GradientBoostingRegressor(n_estimators=100, random_state=42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.subheader(f"Model Evaluation (Mean Squared Error): {mse:.2f}")
    r_sq= r2_score(y_test, y_pred)
    st.subheader(f"R-squared Error:{r_sq:.2f}")
    st.subheader("Model Predictions vs Actual Values:")
    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.write(results)
    st.subheader("Train vs Test Performance:")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.index, y_test, label="Actual", alpha=0.7)
    plt.scatter(y_test.index, y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted on Test Set")
    plt.xlabel("Data Points")
    plt.ylabel(target_variable)
    plt.legend()
    st.pyplot()
    

# Model Prediction Page
elif page == "SocialEdu":
    st.subheader("Model Prediction")
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        target_variable = st.selectbox("Select the target variable", data.columns)
        feature_columns = st.multiselect("Select the input features", data.columns[:-1])
        X = data[feature_columns]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.subheader("Select a model:")
        model_name = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"])
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor()
        elif model_name == "XGBoost":
            model = XGBRegressor()
        elif model_name == "Neural Network":
            model = MLPRegressor(max_iter=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.subheader(f"Model Evaluation (Mean Squared Error): {mse:.2f}")
        r_sq= r2_score(y_test, y_pred)
        st.subheader(f"R-squared Error:{r_sq:.2f}")
        st.subheader("Model Predictions vs Actual Values:")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write(results)
        st.subheader("Train vs Test Performance:")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.index, y_test, label="Actual", alpha=0.7)
        plt.scatter(y_test.index, y_pred, label="Predicted", alpha=0.7)
        plt.title("Actual vs Predicted on Test Set")
        plt.xlabel("Data Points")
        plt.ylabel(target_variable)
        plt.legend()
        st.pyplot()
elif page=="EduSip":
    st.subheader("Data Exploration")
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.subheader("Data Description:")
        st.text(data_description)
        st.subheader("Sample Data:")
        st.write(data.head())

        st.subheader("Correlation Matrix:")
        correlation_matrix = data.corr()



        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix[['G3']], annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix for Final Grade (G3)')
        plt.show()
        st.subheader("Model Prediction")
    # uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    # if uploaded_file is not None:
        # data = pd.read_csv(uploaded_file)
        target_variable = st.selectbox("Select the target variable", data.columns)
        feature_columns = st.multiselect("Select the input features", data.columns[:-1])
        X = data[feature_columns]
        y = data[target_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.subheader("Select a model:")
        model_name = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network","Gradient Boosing",])
        if model_name == "Linear Regression":
            model = LinearRegression()
        elif model_name == "Random Forest":
            model = RandomForestRegressor()
        elif model_name == "XGBoost":
            model = XGBRegressor()
        elif model_name == "Neural Network":
            model = MLPRegressor(max_iter=100)
        elif model_name=="Gradient boosting":
            model= GradientBoostingRegressor(n_estimators=100, random_state=42)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        st.subheader(f"Model Evaluation (Mean Squared Error): {mse:.2f}")
        r_sq= r2_score(y_test, y_pred)
        st.subheader(f"R-squared Error:{r_sq:.2f}")
        st.subheader("Model Predictions vs Actual Values:")
        results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        st.write(results)
        st.subheader("Train vs Test Performance:")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test.index, y_test, label="Actual", alpha=0.7)
        plt.scatter(y_test.index, y_pred, label="Predicted", alpha=0.7)
        plt.title("Actual vs Predicted on Test Set")
        plt.xlabel("Data Points")
        plt.ylabel(target_variable)
        plt.legend()
        st.pyplot()
        

# Model Prediction Page
elif page == "SocialEdu":
    st.subheader("Model Prediction")
    
    uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    target_variable = st.selectbox("Select the target variable", data.columns)
    feature_columns = st.multiselect("Select the input features", data.columns[:-1])
    X = data[feature_columns]
    y = data[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.subheader("Select a model:")
    model_name = st.selectbox("Select a model", ["Linear Regression", "Random Forest", "XGBoost", "Neural Network"])
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Random Forest":
        model = RandomForestRegressor()
    elif model_name == "XGBoost":
        model = XGBRegressor()
    elif model_name == "Neural Network":
        model = MLPRegressor(max_iter=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.subheader(f"Model Evaluation (Mean Squared Error): {mse:.2f}")
    r_sq= r2_score(y_test, y_pred)
    st.subheader(f"R-squared Error:{r_sq:.2f}")
    st.subheader("Model Predictions vs Actual Values:")
    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.write(results)
    st.subheader("Train vs Test Performance:")
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test.index, y_test, label="Actual", alpha=0.7)
    plt.scatter(y_test.index, y_pred, label="Predicted", alpha=0.7)
    plt.title("Actual vs Predicted on Test Set")
    plt.xlabel("Data Points")
    plt.ylabel(target_variable)
    plt.legend()
    st.pyplot()
