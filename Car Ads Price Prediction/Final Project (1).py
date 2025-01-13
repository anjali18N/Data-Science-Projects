#!/usr/bin/env python
# coding: utf-8

# In[7]:


""" 
Final Project Topic 3

Anjali Neupane

The Rdata file was converted into csv in the Rstudio for easy access of the data in Python script.
# Load the RData file
load("C:/Users/neupa/Downloads/car_ads_fp.RData")

# List all objects in the R session to find the dataset name
ls()

# write it to a CSV file
write.csv(carAd, "C:/Users/neupa/Downloads/car_ads.csv", row.names = FALSE)"""


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load the converted dataset
df = pd.read_csv('c:/Users/neupa/OneDrive/Desktop/Python Programming/car_ads.csv')

# Display the first few rows of the dataframe to understand its structure
df.head()

# Filter the dataset for the specified vehicle models, body types, and fuel type
filtered_df = df[(df['Genmodel'].isin(['L200', 'XC90', 'Sorento', 'Outlander'])) &
                 (df['Bodytype'].isin(['SUV', 'Pickup'])) &  
                 (df['Fuel_type'] == 'Diesel')]

# Identify the six most frequently advertised colors
top_colors = filtered_df['Color'].value_counts().head(6).index.tolist()

# Further filter the dataset for the top colors
filtered_df = filtered_df[filtered_df['Color'].isin(top_colors)]

# Check the shape of the filtered dataset to ensure it matches the expected structure
print(f"Filtered dataset shape: {filtered_df.shape}")

# Display the first few rows to verify the filtering
filtered_df.head()



# Selecting features and target variable
features = ['Maker', 'Genmodel', 'Color', 'Bodytype', 'Gearbox', 'Reg_year', 'Runned_Miles', 'Engin_size']
target = 'Price'

# Encoding categorical variables
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(filtered_df[features].astype(str))  # Ensure all categorical data are strings for encoding
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(features))

# Numerical features are directly taken from the dataframe
numerical_features_df = filtered_df[['Reg_year', 'Runned_Miles']].reset_index(drop=True)

# Combine encoded and numerical features
X = pd.concat([encoded_features_df, numerical_features_df], axis=1)
y = filtered_df[target].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model fitting
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model evaluation
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
training_r2 = r2_score(y_train, y_pred_train)
testing_r2 = r2_score(y_test, y_pred_test)
training_rmse = mean_squared_error(y_train, y_pred_train, squared=False)
testing_rmse = mean_squared_error(y_test, y_pred_test, squared=False)

# Display the evaluation metrics
print(f"Training R²: {training_r2:.3f}")
print(f"Testing R²: {testing_r2:.3f}")
print(f"Training RMSE: {training_rmse:.2f}")
print(f"Testing RMSE: {testing_rmse:.2f}")

# Extracting feature importances
feature_importances = model.feature_importances_
features_list = X.columns  # List of features

# Creating a DataFrame for easier visualization
importances_df = pd.DataFrame({'Feature': features_list, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False).reset_index(drop=True)

# Display the top 10 most important features
importances_df.head(10)





# In[ ]:




