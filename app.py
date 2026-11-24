"""
Real Estate Valuation - Streamlit App
Run locally:  streamlit run app.py
"""

import pickle

import pandas as pd
import streamlit as st

# Same feature order that was used while training (see train_model.py)
FEATURES = [
    "transaction_date",
    "house_age",
    "distance_to_mrt",
    "convenience_stores",
    "latitude",
    "longitude",
]

st.set_page_config(page_title="House Price Predictor", page_icon="🏠")


@st.cache_resource
def load_model():
    """Load the saved pipeline once (no retraining here)."""
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()

# ---------------- Title and description ----------------
st.title("🏠 House Price Predictor")
st.write(
    "This app estimates the **house price per unit area** for a property in "
    "the Sindian district of New Taipei City, Taiwan. It uses a Random Forest "
    "model trained on the UCI Real Estate Valuation dataset (414 sales from "
    "2012-2013)."
)

# ---------------- Input controls ----------------
st.subheader("Enter the property details")

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Year of sale", [2012, 2013], index=1)
    month = st.selectbox("Month of sale", list(range(1, 13)), index=5)
    house_age = st.number_input(
        "House age (years)", min_value=0.0, max_value=50.0, value=10.0, step=0.5
    )
    stores = st.slider("Convenience stores within walking distance", 0, 10, 4)

with col2:
    distance = st.number_input(
        "Distance to nearest MRT (metro) station, in metres",
        min_value=0.0,
        max_value=7000.0,
        value=500.0,
        step=50.0,
    )
    latitude = st.number_input(
        "Latitude", min_value=24.93, max_value=25.02, value=24.97, format="%.5f"
    )
    longitude = st.number_input(
        "Longitude", min_value=121.47, max_value=121.57, value=121.53, format="%.5f"
    )

# ---------------- Prediction ----------------
if st.button("Predict price", type="primary"):
    # Basic input validation
    errors = []
    if (year, month) < (2012, 8) or (year, month) > (2013, 7):
        errors.append(
            "The model has only seen sales from August 2012 to July 2013. "
            "Please choose a date in that range."
        )
    if errors:
        for e in errors:
            st.error(e)
    else:
        # The dataset stores the date as a decimal year, e.g. 2013.5 = June 2013
        transaction_date = year + month / 12

        input_df = pd.DataFrame(
            [[transaction_date, house_age, distance, stores, latitude, longitude]],
            columns=FEATURES,
        )
        price = float(model.predict(input_df)[0])

        st.success(f"Estimated price: **{price:.1f} × 10,000 NT$ per Ping**")
        st.write(
            f"That is about **NT$ {price * 10000:,.0f} per Ping** "
            f"(1 Ping = 3.3 m²), or roughly "
            f"**NT$ {price * 10000 / 3.3:,.0f} per square metre**."
        )
        st.caption(
            "On unseen test data the model was wrong by about 3.7 units "
            "(NT$ 37,000 per Ping) on average, so treat this as a rough estimate."
        )

        with st.expander("Show the values sent to the model"):
            st.dataframe(input_df, hide_index=True)

# ---------------- Limitations ----------------
st.divider()
st.warning(
    "**Limitations:** This is an academic prototype made for a college mini "
    "project. It is trained on only 414 sales from one district of Taiwan in "
    "2012-2013, so it does not reflect current prices or other locations. "
    "Do not use it for real buying, selling, or financial decisions."
)
