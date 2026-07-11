# Airline Passenger Satisfaction Prediction

Predicting whether airline passengers are satisfied or dissatisfied based on their survey responses, flight details, and demographics.

## Project Overview

This project uses the Invistico Airline dataset (~130k passenger survey responses) to explore what actually drives passenger satisfaction, and to build a model that predicts it. The goal isn't just a working model, it's understanding why passengers feel the way they do, so the findings could realistically inform where an airline should focus its efforts.


## Workflow

```
01_data_cleaning_EDA.ipynb          - airline_satisfaction_clean.csv
02_feature_engineering.ipynb        - engineered feature set
03_model_training_evaluation.ipynb  - trained_model.pkl 
04_interpretation.ipynb             - permutation importance + recommendations
```
**Note:** models/trained_model.pkl is not included in this repository due to its size (over GitHub's 100MB limit). Run 03_model_training_evaluation.ipynb to
regenerate it locally, the random seed is fixed, so results can be reproduced.

## Key Findings

### Data Cleaning & EDA (01_data_cleaning_eda.ipynb)

- **Target balance:** satisfaction is fairly balanced about 55% satisfied, 45% dissatisfied so accuracy is a reasonable metric here, no class-imbalance workarounds needed.

- **Strongest service drivers:** inflight entertainment, ease of online booking, and online support showed the clearest gap between satisfied and dissatisfied passengers, both in raw rating comparisons and in correlation with the target.

- **Weakest signals:** gate location and departure/arrival time convenience barely separated satisfied from dissatisfied passengers at all. Flight distance was weak too.

- **Delays mattered less than expected:** despite the common assumption that flight delays drive dissatisfaction, both delay columns showed only a weak correlation with satisfaction (around -0.07 to -0.08) the service experience mattered far more than how late the flight was.

- **Class and travel type mattered:** Business class and Business travel both leaned satisfied; Economy and Personal travel leaned dissatisfied. Customer type (loyal vs. disloyal) showed an even bigger split.

- Departure delay and arrival delay were almost perfectly correlated with each other (0.97), a strong signal that they were carrying largely redundant information.

### Feature Engineering (02_feature_engineering.ipynb)

- Combined the two delay columns into a single total_delay feature, given how redundant they were.

- Added avg_service_score, a composite average across all 14 service ratings. Correlation with satisfaction came out to 0.51, right on par with the single strongest individual service rating, confirming the composite feature captured real signal rather than diluting it.

- total_delay correlated with satisfaction at -0.078, consistent with the individual delay columns from the EDA confirming delays just aren't a major driver here.

### Model Training & Evaluation (03_model_training_evaluation.ipynb)

- **Random Forest substantially outperformed Logistic Regression** 99.3% ROC-AUC and 96% accuracy, compared to Logistic Regression's 90.8% ROC-AUC and 83% accuracy. This gap was much larger than expected going in, and held up consistently across both cross-validation and the held-out test set.

- Both models handled the two classes evenly (no lopsided precision/recall), so the default 0.5 decision threshold was used as-is, no threshold tuning needed.

- Random Forest was saved as the final model.

### Model Interpretation (04_interpretation.ipynb)

- **Found the explanation for the performance gap:** the two strongest predictive features **Seat comfort** and **Inflight entertainment** both show a U-shaped, non-monotonic relationship with satisfaction. A rating of 0 behaves like a rating of 5 (high satisfaction), while ratings 1-3 correspond to low satisfaction. This strongly suggests a rating of 0 actually means "didn't use this service," not "worst possible experience." A linear model structurally cannot represent this kind of pattern, no matter how much data it sees. Which explains a real chunk of why Random Forest won by such a wide margin.

- **Online support** showed a threshold effect rather than a U-shape, satisfaction stays flat through ratings 1-3, then jumps sharply at a rating of 4. The real opportunity there is pushing scores from "good" to "great," not from "poor" to "mediocre."

- **Customer Type, Type of Travel, and Gender** all showed clean, roughly monotonic relationships with satisfaction, features a linear model has no trouble with.

- **Permutation importance corrected the earlier, less reliable ranking:** Seat comfort turned out to be the dominant feature by a wide margin (not Inflight entertainment, as Random Forest's built-in impurity-based importance had suggested), and Customer Type and Type of Travel mattered more than that quick-look ranking indicated.

## Business Recommendations

1. **Investigate what a rating of 0 actually means** for Seat comfort and Inflight entertainment. If it represents "not applicable" rather than a genuine 0-5 rating, future surveys should separate that out, conflating the two is misleading anyone reading these ratings as a simple average.

2. **For Online support, focus on pushing ratings from 3 to 4**, not from 1 to 2 that's where nearly all the satisfaction gain is concentrated.

3. **Loyalty and travel type are clean, reliable targets** for retention-style efforts, since neither shows the non-monotonic complications the top service ratings do.

4. **Don't prioritize gate location or departure/arrival time convenience** both showed almost no relationship with satisfaction across every stage of this analysis.

## Dataset

- **Source:** Invistico Airline passenger satisfaction survey
- **Size:** 129,880 rows, 23 columns (129,487 after dropping ~0.3% missing values in Arrival Delay in Minutes)
- **Target:** satisfaction (satisfied / dissatisfied)
- **Features:** demographics (age, gender, customer type, type of travel, class), flight distance, 14 service-quality ratings (0–5 scale), and two delay measurements

## Repository Structure

```
Airline Satisfaction Prediction/
├── README.md
├── data/
│   ├── raw/
│   │   └── Invistico_Airline.csv
│   └── processed/
│       ├── airline_satisfaction_clean.csv
│       ├── airline_satisfaction_features.csv
│       └── airline_satisfaction_features_scaled.csv
├── models/
│   └── trained_model.pkl
├── images/                       # Charts saved from each notebook
└── notebooks/
    ├── 01_data_cleaning_eda.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_model_training_evaluation.ipynb
    └── 04_interpretation.ipynb
```


## Limitations

- **No causal claims** permutation importance shows what the model relies on, not necessarily what causes satisfaction in the real world.
- **Survey data has its own biases** respondents may not represent all passengers equally, and self-reported ratings can be influenced by overall mood rather than being fully independent signals.
- **No temporal validation** this is a random train/test split, not a time-based one.
- **Uncalibrated probabilities** Random Forest's predict_proba() output isn't guaranteed to reflect true probabilities without calibration.
- **Limited scope** this dataset covers a single airline's survey responses; findings may not generalize elsewhere.

## Future Work

- Investigate whether a rating of 0 genuinely means "not applicable" in this dataset, and re-encode it as a separate flag rather than treating it as part of the 0-5 scale, this could meaningfully close the performance gap for Logistic Regression.
- Try a boosting model (e.g. XGBoost) to see if it matches or beats Random Forest, and whether it highlights the same top features.
- Calibrate the model's probabilities so they could be used directly, rather than only for ranking.
- A/B test the top business recommendation to confirm it actually moves satisfaction in practice.
