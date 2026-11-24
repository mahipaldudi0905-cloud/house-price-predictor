"""
Real Estate Valuation - Model Training Script
----------------------------------------------
Run:  python train_model.py
Output: model.pkl  (the complete fitted Pipeline: scaling + model)

Dataset: UCI Real Estate Valuation (414 records, New Taipei City, Taiwan)
Task:    Regression - predict house price per unit area
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42

# The order of features used everywhere (training AND the Streamlit app)
FEATURES = [
    "transaction_date",
    "house_age",
    "distance_to_mrt",
    "convenience_stores",
    "latitude",
    "longitude",
]
TARGET = "price_per_unit_area"


def load_data(path="data/dataset.csv"):
    """Load the CSV and give the columns short, easy names."""
    df = pd.read_csv(path)
    df.columns = ["No"] + FEATURES + [TARGET]
    df = df.drop(columns=["No"])  # 'No' is only a serial number, not useful
    return df


def main():
    # ---------------- 1. Load data ----------------
    df = load_data()
    print("Rows, Columns:", df.shape)
    print("Missing values:", int(df.isna().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))

    X = df[FEATURES]
    y = df[TARGET]

    # ---------------- 2. Train-test split ----------------
    # 80% for training/tuning, 20% kept aside for the final test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])

    # ---------------- 3. Compare models with 5-fold cross-validation ----------------
    # Every model is inside a Pipeline, so scaling is learned only from
    # the training folds (no information leaks from validation/test data).
    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(random_state=RANDOM_STATE),
    }

    print("\n--- 5-fold cross-validation on TRAINING data ---")
    rows = []
    for name, model in models.items():
        pipe = Pipeline([("scaler", StandardScaler()), ("model", model)])
        r2 = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")
        rmse = -cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="neg_root_mean_squared_error"
        )
        mae = -cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error"
        )
        rows.append([name, mae.mean(), rmse.mean(), r2.mean(), r2.std()])

    results = pd.DataFrame(
        rows, columns=["Model", "CV MAE", "CV RMSE", "CV R2", "R2 std"]
    ).sort_values("CV RMSE")
    print(results.round(3).to_string(index=False))

    # ---------------- 4. Hyperparameter tuning (GridSearchCV) ----------------
    # Random Forest gave the best cross-validation result, so we tune it.
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    }
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print("\n--- GridSearchCV ---")
    print("Best parameters:", grid.best_params_)
    print("Best CV RMSE   :", round(-grid.best_score_, 3))

    # ---------------- 5. Final test (done only ONCE, at the end) ----------------
    best_pipeline = grid.best_estimator_
    y_pred = best_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Final result on unseen TEST data ---")
    print("MAE :", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print("R2  :", round(r2, 3))

    # ---------------- 6. Save the complete pipeline ----------------
    with open("model.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
    print("\nSaved the fitted pipeline to model.pkl")


if __name__ == "__main__":
    main()
