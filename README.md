# 🏠 Real Estate Valuation — House Price Prediction

A machine learning mini project that predicts the **house price per unit area** of a property and serves the prediction through a Streamlit web app.

> **Academic prototype.** Trained on 414 sales from one district of Taiwan (2012–2013). Not for real buying, selling or financial decisions.

## Links

- **Deployed app:** https://house-price-predictor-c4dvuqicgrewj5hdxff4nn.streamlit.app/
- **GitHub repository:** https://github.com/mahipaldudi0905-cloud/house-price-predictor

## Problem statement

| Question | Answer |
|---|---|
| What is predicted? | House price per unit area (10,000 New Taiwan Dollars per Ping; 1 Ping = 3.3 m²) |
| Task type | Regression |
| Input features | Transaction date, house age, distance to nearest MRT station, number of convenience stores, latitude, longitude |
| Who could use it? | A buyer or seller who wants a rough estimate of a fair price |
| Limitation | Small, old dataset from one location; estimate only |

## Dataset

- **Name:** Real Estate Valuation
- **Source:** UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set
- **Citation:** Yeh, I-C. (2018). Real Estate Valuation [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5J30W
- **Size:** 414 records, 6 features, 1 target. All columns are numerical. No missing values, no duplicate rows.

| Column | Meaning |
|---|---|
| `transaction_date` | Date of sale as a decimal year (2013.250 = March 2013) |
| `house_age` | Age of the house in years |
| `distance_to_mrt` | Distance to the nearest metro station in metres |
| `convenience_stores` | Number of convenience stores within walking distance |
| `latitude`, `longitude` | Location of the house |
| `price_per_unit_area` | **Target** — price in 10,000 NT$ per Ping |

## Method

1. **EDA** — distributions, correlation heatmap, feature-vs-price plots (`notebooks/analysis.ipynb`)
2. **Preprocessing** — removed the serial-number column; `StandardScaler` inside a `Pipeline`
3. **Train-test split** — 80% train (331 rows), 20% test (83 rows)
4. **Model comparison** — 4 models with 5-fold cross-validation on the training data
5. **Hyperparameter tuning** — `GridSearchCV` on the best model (Random Forest)
6. **Final test** — one evaluation on the held-out test set
7. **Saving** — the complete fitted pipeline is pickled to `model.pkl`
8. **App** — `app.py` loads `model.pkl` and predicts (no retraining)

## Results

**5-fold cross-validation on training data**

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Random Forest** | **5.17** | **8.14** | **0.63** |
| KNN Regressor | 6.03 | 8.80 | 0.58 |
| Linear Regression | 6.45 | 9.18 | 0.54 |
| Decision Tree | 6.63 | 10.79 | 0.30 |

**Tuning (GridSearchCV, 27 combinations × 5 folds)** — best: `n_estimators=300`, `max_depth=None`, `min_samples_leaf=4`, CV RMSE 7.84.

**Final tuned Random Forest on the test set**

| MAE | RMSE | R² |
|---|---|---|
| 3.67 | 5.46 | 0.82 |

Random Forest was chosen because it had the lowest error in cross-validation, handles the curved relation between metro distance and price, and is more stable than a single decision tree.

## Project structure

```
ML_Mini_Project/
├── data/
│   └── dataset.csv
├── notebooks/
│   └── analysis.ipynb
├── train_model.py
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run locally

```bash
pip install -r requirements.txt
python train_model.py        # optional: retrains and recreates model.pkl
streamlit run app.py
```

## Limitations

- Only 414 records from the Sindian district of New Taipei City.
- Data is from August 2012 to July 2013; prices today are different.
- One unusually expensive sale (117.5) affects the error.
- The app is an educational prototype, not a valuation tool.

## Team

| Member | Contribution |
|---|---|
| Mahipal | Problem definition, EDA, model building, pipeline, Streamlit app, deployment |
