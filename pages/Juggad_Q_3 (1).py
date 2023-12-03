#!/usr/bin/env python
# coding: utf-8

# In[66]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import streamlit as st


# In[67]:


data=pd.read_csv("\encoded.csv")


# In[68]:


data


# In[69]:


# Define the features and the target
X = data[['Mjob', 'Fjob']] # Including some more features that might affect the social life
y = data['G3']


# In[70]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# # Random Forest Regressor

# In[71]:


# Initialize the RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)


# In[72]:


# Fit the model
rf.fit(X_train, y_train)


# In[73]:


# Predict on the test set
y_pred = rf.predict(X_test)


# In[74]:


# Calculate the root mean squared error
rmse = sqrt(mean_squared_error(y_test, y_pred))


# In[75]:


# Output the RMSE
st.write('Root Mean Squared Error:', rmse)


# In[76]:


r_squared = r2_score(y_test, predictions)
st.write(r_squared)


# In[77]:


# Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')


# In[78]:


from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
r_squared = r2_score(y_test, y_pred)
st.write(r_squared)


# # Gradient Boosting Regressor

# In[79]:


from sklearn.ensemble import GradientBoostingRegressor


# In[80]:


# Initialize the GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)


# In[81]:


# Fit the model

from tqdm.auto import tqdm
with tqdm(total=100) as pbar:
    gbr.fit(X_train, y_train)
    pbar.update(50)


# In[82]:


# Predict on the test set
y_pred_gbr = gbr.predict(X_test)
pbar.update(30)


# In[83]:


# Calculate the root mean squared error
rmse_gbr = sqrt(mean_squared_error(y_test, y_pred_gbr))
pbar.update(20)


# In[84]:


# Output the RMSE
st.write('Root Mean Squared Error for Gradient Boosting Regressor:', rmse_gbr)


# In[85]:


r_squared = r2_score(y_test, y_pred_gbr)
st.write(r_squared)


# In[86]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Assuming your DataFrame is named 'data'
# Replace 'data.csv' with your actual data file if needed
data = pd.read_csv('E:\\Hackathon 23\\encoded.csv')

# Selecting features related to social life
social_features = ['Mjob', 'Fjob']

# Extracting the relevant features and target variable
X_social = data[social_features]
y = data['G3']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_social, y, test_size=0.2, random_state=456)

# Train a GradientBoostingRegressor model
model = GradientBoostingRegressor(n_estimators=100, random_state=456)
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r_squared = r2_score(y_test, predictions)

st.write(f'Mean Squared Error: {mse}')
st.write(f'R-squared: {r_squared}')


# # Linear Regression

# In[87]:


# Train a Linear Regression model with selected features
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[88]:


# # # Make predictions
predictions = lr_model.predict(X_test)


# In[89]:


# # Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')


# In[90]:


r_squared = r2_score(y_test, predictions)
st.write(r_squared)


# # XG-Boost

# In[91]:


from xgboost import XGBRegressor # Training the XGBoost Regressor
model = XGBRegressor(n_estimators=100, random_state=456)  # You can adjust the parameters as needed
model.fit(X_train, y_train)


# In[92]:


# Make predictions
predictions = model.predict(X_test)


# In[93]:


# Evaluate the model
mse = mean_squared_error(y_test, predictions)
st.write(f'Mean Squared Error: {mse}')


# In[94]:


r_squared = r2_score(y_test, predictions)
st.write(f'R_squared value is: {r_squared}')


# In[ ]:





# In[ ]:




