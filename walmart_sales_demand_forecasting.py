"""Walmart Sales Demand Forecasting

Educational machine-learning project based on my final project.
Expected dataset: Walmart.csv
Target: Weekly_Sales
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

RANDOM_STATE = 42
DATA_PATH = "Walmart.csv"
TARGET = "Weekly_Sales"

sns.set_theme(style="whitegrid")


def evaluate(model_name, model, X, y):
    """Return a consistent metric row for a fitted regression model."""
    pred = model.predict(X)
    return {
        "Model": model_name,
        "RMSE": np.sqrt(mean_squared_error(y, pred)),
        "MAE": mean_absolute_error(y, pred),
        "R2": r2_score(y, pred),
    }


def add_date_features(df):
    """Convert Date into numeric calendar features used for modeling."""
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            format="%d-%m-%Y",
            errors="coerce",
        )
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype("Int64")

    return df


def main():
    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in {DATA_PATH}")

    print("Dataset shape:", df.shape)
    print("\nMissing values:\n", df.isna().sum())

    # Exploratory analysis
    plt.figure(figsize=(7, 4))
    sns.histplot(df[TARGET], bins=30, kde=True)
    plt.title("Weekly Sales Distribution")
    plt.tight_layout()
    plt.show()

    numeric = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(11, 8))
    sns.heatmap(numeric.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    eda = add_date_features(df)

    if "Date" in eda.columns and eda["Date"].notna().any():
        ordered = eda.sort_values("Date")
        plt.figure(figsize=(12, 4))
        plt.plot(ordered["Date"], ordered[TARGET])
        plt.title("Weekly Sales Trend Over Time")
        plt.xlabel("Date")
        plt.ylabel(TARGET)
        plt.tight_layout()
        plt.show()

    if "Month" in eda.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=eda, x="Month", y=TARGET)
        plt.title("Weekly Sales by Month")
        plt.tight_layout()
        plt.show()

    if "Holiday_Flag" in eda.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=eda, x="Holiday_Flag", y=TARGET)
        plt.title("Weekly Sales by Holiday Flag")
        plt.tight_layout()
        plt.show()

    model_df = add_date_features(df)

    if "Date" in model_df.columns:
        model_df = model_df.drop(columns=["Date"])

    y = model_df[TARGET].copy()
    X = model_df.drop(columns=[TARGET]).copy()

    has_time_features = all(
        col in X.columns for col in ["Year", "Month", "WeekOfYear"]
    )

    if has_time_features:
        combined = pd.concat([X, y], axis=1)
        combined = combined.sort_values(["Year", "Month", "WeekOfYear"])
        split = int(len(combined) * 0.80)

        train = combined.iloc[:split]
        test = combined.iloc[split:]

        X_train = train.drop(columns=[TARGET])
        y_train = train[TARGET]
        X_test = test.drop(columns=[TARGET])
        y_test = test[TARGET]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    models = {
        "LinearRegression (baseline)": Pipeline(
            [("prep", preprocess), ("model", LinearRegression())]
        ),
        "KNNRegressor (baseline)": Pipeline(
            [("prep", preprocess), ("model", KNeighborsRegressor())]
        ),
        "SVR (baseline)": Pipeline(
            [("prep", preprocess), ("model", SVR())]
        ),
        "RandomForestRegressor (baseline)": Pipeline(
            [
                ("prep", preprocess),
                ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
            ]
        ),
        "GradientBoostingRegressor (baseline)": Pipeline(
            [
                ("prep", preprocess),
                ("model", GradientBoostingRegressor(random_state=RANDOM_STATE)),
            ]
        ),
    }

    baseline_rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        baseline_rows.append(evaluate(name, model, X_test, y_test))

    baseline = pd.DataFrame(baseline_rows).sort_values("RMSE")
    print("\nBaseline leaderboard:\n")
    print(baseline.to_string(index=False))

    cv = TimeSeriesSplit(n_splits=5) if has_time_features else 5

    rf_search = RandomizedSearchCV(
        estimator=models["RandomForestRegressor (baseline)"],
        param_distributions={
            "model__n_estimators": [200, 400, 800],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.6, 1.0],
        },
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    svr_search = RandomizedSearchCV(
        estimator=models["SVR (baseline)"],
        param_distributions={
            "model__C": np.logspace(-1, 3, 20),
            "model__gamma": np.logspace(-4, 0, 20),
            "model__epsilon": [0.01, 0.05, 0.1, 0.2],
            "model__kernel": ["rbf"],
        },
        n_iter=25,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    svr_search.fit(X_train, y_train)

    tuned = pd.DataFrame(
        [
            evaluate(
                "RandomForestRegressor (tuned)",
                rf_search.best_estimator_,
                X_test,
                y_test,
            ),
            evaluate(
                "SVR (tuned)",
                svr_search.best_estimator_,
                X_test,
                y_test,
            ),
        ]
    )

    leaderboard = pd.concat([baseline, tuned], ignore_index=True).sort_values("RMSE")
    print("\nFinal leaderboard:\n")
    print(leaderboard.to_string(index=False))

    best_model = rf_search.best_estimator_
    pred = best_model.predict(X_test)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, alpha=0.6)
    low = min(y_test.min(), pred.min())
    high = max(y_test.max(), pred.max())
    plt.plot([low, high], [low, high], linestyle="--")
    plt.title("Actual vs Predicted Weekly Sales")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.tight_layout()
    plt.show()

    residuals = y_test.to_numpy() - pred
    plt.figure(figsize=(9, 4))
    sns.histplot(residuals, bins=30, kde=True)
    plt.axvline(0, linestyle="--")
    plt.title("Residual Distribution")
    plt.xlabel("Actual - Predicted")
    plt.tight_layout()
    plt.show()

    perm = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring="neg_root_mean_squared_error",
    )

    feature_names = best_model.named_steps["prep"].get_feature_names_out()
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    print("\nPermutation importance:\n")
    print(importance.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
