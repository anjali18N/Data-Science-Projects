# Airline Passenger Satisfaction Prediction

Predicting whether airline passengers are satisfied or dissatisfied based on their survey responses, flight details, and demographics.

## Project Overview

This project uses the Invistico Airline dataset (~130k passenger survey responses) to explore what actually drives passenger satisfaction, and to build a model 
that predicts it. The goal isn't just a working model, it's understanding why passengers feel the way they do, so the findings could realistically inform where 
an airline should focus its efforts.

**Status: in progress.** Data cleaning and EDA are done; feature engineering, modeling, and evaluation need some changes.

## Dataset

- **Source:** Invistico Airline passenger satisfaction survey
- **Size:** 129,880 rows, 23 columns
- **Target:** `satisfaction` (satisfied / dissatisfied)
- **Features:** demographics (age, gender, customer type, type of travel, class), flight distance, 14 service-quality ratings (0–5 scale), and two delay 
measurements

## Repository Structure

```
Airline Satisfaction Prediction/
├── README.md
├── data/
│   ├── raw/
│   │   └── Invistico_Airline.csv
│   └── processed/
│       └── airline_satisfaction_clean.csv
├── images/                       # Charts saved from each notebook
└── notebooks/
    └── 01_data_cleaning_eda.ipynb
```
