# Walmart Sales Demand Forecasting

A machine learning regression project for forecasting **Weekly_Sales** using Walmart retail data.

This project was developed from my machine learning final project and demonstrates feature engineering, time-aware train/test splitting, model comparison, hyperparameter tuning, error analysis, and feature importance.

## Project Highlights

- Engineered date features including Year, Month, Day, and WeekOfYear
- Used a time-aware 80/20 split when date information was available
- Compared 5 regression approaches
- Tuned Random Forest and SVR with `RandomizedSearchCV`
- Evaluated models using RMSE, MAE, and R²
- Used permutation importance to interpret the final model
- Analyzed seasonal, holiday, store, and economic factors

## Models Compared

- Linear Regression
- K-Nearest Neighbors Regressor
- Support Vector Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Best Model

The tuned Random Forest produced the strongest performance in the completed notebook run.

| Metric | Score |
|---|---:|
| RMSE | 164,357.75 |
| MAE | 93,374.34 |
| R² | 0.9045 |

The baseline Random Forest was also strong with an R² of approximately **0.9025**, while Gradient Boosting reached approximately **0.8644**.

## Technologies

Python · Pandas · NumPy · Scikit-learn · Matplotlib · Seaborn

## Repository Files

- `walmart_sales_demand_forecasting.py` — cleaned forecasting workflow
- `requirements.txt` — Python dependencies

The script expects a dataset named `Walmart.csv`.

## Run

```bash
pip install -r requirements.txt
python walmart_sales_demand_forecasting.py
```

## Skills Demonstrated

Regression · Demand Forecasting · Feature Engineering · Time-Aware Validation · Hyperparameter Tuning · Model Evaluation · Feature Importance · Python
