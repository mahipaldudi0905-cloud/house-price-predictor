"""
Quick manual test - predict the price for any custom property values.
Run:  python test_model.py
"""

import pickle

import pandas as pd

FEATURES = [
    "transaction_date",
    "house_age",
    "distance_to_mrt",
    "convenience_stores",
    "latitude",
    "longitude",
]

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# ---- Change these numbers to test any property you like ----
sample = {
    "transaction_date": 2013.25,   # e.g. 2013.25 = March 2013
    "house_age": 15.0,             # years
    "distance_to_mrt": 500.0,      # metres to nearest MRT station
    "convenience_stores": 5,       # count
    "latitude": 24.97,
    "longitude": 121.53,
}

input_df = pd.DataFrame([sample], columns=FEATURES)
price = model.predict(input_df)[0]

print("Input:", sample)
print(f"Predicted price: {price:.2f} (x 10,000 NT$ per Ping)")
