```python
Car Ads Price Prediction

This project analyzes car advertisements and predicts car prices using machine learning techniques. The dataset was originally in RData format and was converted to CSV for easier access in Python. The goal of this project is to predict car prices based on various features like car model, color, body type, and others.

Objective
The objective of this project is to filter a car advertisement dataset, preprocess the data, and build a machine learning model to predict car prices based on features such as the maker, model, color, body type, gearbox, and others.

Dataset
The dataset contains the following features:

Maker: The manufacturer of the car.
Genmodel: The specific model of the car.
Color: The color of the car.
Bodytype: The type of the car (e.g., SUV, Pickup).
Gearbox: The type of gearbox (e.g., Manual, Automatic).
Reg_year: The registration year of the car.
Runned_Miles: The number of miles the car has run.
Engin_size: The size of the car engine.
Fuel_type: The type of fuel used (e.g., Diesel).
Price: The target variable, representing the price of the car.

Methods
Data Preprocessing:

Converted the RData file to CSV for easy use in Python.
Filtered the dataset to focus on specific car models, body types, and fuel type (Diesel).
Further filtered to include only the top 6 most frequently advertised car colors.

Feature Engineering:

Used one-hot encoding to convert categorical features (Maker, Genmodel, Color, Bodytype, Gearbox) into numerical features.
Used numerical features like Reg_year, Runned_Miles, and Engin_size directly for modeling.

Modeling:

Built a predictive model using Extra Trees Regressor from sklearn.
Split the dataset into training and testing sets for model evaluation.
Evaluated the model using R² and RMSE metrics.

Evaluation
Training R²: Measures the proportion of variance in the training data explained by the model.
Testing R²: Measures the proportion of variance in the testing data explained by the model.
Training RMSE: Root Mean Squared Error on the training data.
Testing RMSE: Root Mean Squared Error on the testing data.

Results
The model performed well, with the following evaluation metrics:

Training R²: 0.999
Testing R²: 0.961
Training RMSE: 283.80
Testing RMSE: 2246.65

Files in This Repository
original RData file
car_ads_price_prediction.py: The Python script for data preprocessing, feature engineering, model training, and evaluation.
README.md: The project documentation.

Requirements
To run the Python script, install the following libraries:

pip install pandas scikit-learn


```
