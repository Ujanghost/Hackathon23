# -*- coding: utf-8 -*-
"""Juggad Q-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SJcXHuQuEe1IzaQ_2rv1FfDMHUgFIzJt
"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import streamlit as st


data=pd.read_csv("encoded.csv")

data

# Define the features and the target
X = data[['goout', 'Dalc', 'Walc', 'freetime', 'absences']] # Including some more features that might affect the social life
y = data['G3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Calculate the root mean squared error
rmse = sqrt(mean_squared_error(y_test, y_pred))

# Output the RMSE
st.write('Root Mean Squared Error:', rmse)

from sklearn.metrics import r2_score

r_squared = r2_score(y_test, y_pred)
st.write(r_squared)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Scatter plot for actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values (G3)')
plt.ylabel('Predicted Values (G3)')
plt.show()

# Residual plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals, alpha=0.7)
plt.title('Residual Plot')
plt.xlabel('Actual Values (G3)')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # Add a horizontal line at y=0 for reference
plt.show()

# Distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals, bins=30, kde=True)
plt.title('Distribution of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

"""**Gradient Boosting**


"""



from sklearn.ensemble import GradientBoostingRegressor

# Initialize the GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)

# Fit the model

# from tqdm.auto import tqdm
# with tqdm(total=100) as pbar:
gbr.fit(X_train, y_train)
    # pbar.update(50)

# Predict on the test set
y_pred_gbr = gbr.predict(X_test)
# pbar.update(30)

# Calculate the root mean squared error
rmse_gbr = sqrt(mean_squared_error(y_test, y_pred_gbr))
# pbar.update(20)

# Output the RMSE
st.write('Root Mean Squared Error for Gradient Boosting Regressor:', rmse_gbr)

from sklearn.metrics import r2_score
r_squared = r2_score(y_test, y_pred_gbr)
st.write(r_squared)

from sklearn.metrics import r2_score

# Scatter plot for actual vs predicted values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gbr, alpha=0.7)
plt.title('Actual vs Predicted Values (Gradient Boosting Regressor)')
plt.xlabel('Actual Values (G3)')
plt.ylabel('Predicted Values (G3)')
plt.show()

# Residual plot
residuals_gbr = y_test - y_pred_gbr
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=residuals_gbr, alpha=0.7)
plt.title('Residual Plot (Gradient Boosting Regressor)')
plt.xlabel('Actual Values (G3)')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)  # Add a horizontal line at y=0 for reference
plt.show()

# Distribution of residuals
plt.figure(figsize=(10, 6))
sns.histplot(residuals_gbr, bins=30, kde=True)
plt.title('Distribution of Residuals (Gradient Boosting Regressor)')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()

"""Random Forest"""

# Train a model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# # Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

r_squared = r2_score(y_test, predictions)
st.write(r_squared)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score

# Scatter plot for actual vs predicted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=predictions, alpha=0.7)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values (G3)')
plt.ylabel('Predicted Values (G3)')
plt.show()

"""Linear Regression"""

# Train a Linear Regression model with selected features
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# # # Make predictions
predictions = lr_model.predict(X_test)

# # Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

r_squared = r2_score(y_test, predictions)
st.write(r_squared)

"""**XG-Boost**"""

from xgboost import XGBRegressor # Training the XGBoost Regressor
model = XGBRegressor(n_estimators=100, random_state=456)  # You can adjust the parameters as needed
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')

r_squared = r2_score(y_test, predictions)
st.write(r_squared)

