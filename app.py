"""
Streamlit front end for the used-car price model.

Loads the trained pipeline from models/car_price_model.pkl and the brand /
category lists from models/metadata.json (both written by src/train.py) so
the dropdowns always match what the model was actually trained on.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "car_price_model.pkl"
METADATA_PATH = ROOT / "models" / "metadata.json"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH) as f:
        return json.load(f)


def format_inr(amount: float) -> str:
    """Indian digit grouping (5,00,000 instead of 500,000), plus a lakh/crore
    label — how car prices are actually written and read in this market."""
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    s = str(amount)
    if len(s) > 3:
        last3 = s[-3:]
        rest = s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        s = ",".join(groups) + "," + last3
    rupees = f"{sign}₹{s}"
    if amount >= 1_00_00_000:
        label = f" (~{amount / 1_00_00_000:.2f} crore)"
    elif amount >= 1_00_000:
        label = f" (~{amount / 1_00_000:.2f} lakh)"
    else:
        label = ""
    return rupees + label


st.set_page_config(page_title="Used Car Price Estimator", page_icon="\U0001F697",
                    layout="centered")

st.title("Used Car Price Estimator")
st.caption(
    "Trained on ~3,500 cleaned listings from CarDekho (India, data collected "
    "in 2020). Fill in the details below to get an estimated resale price."
)

model = load_model()
meta = load_metadata()

with st.form("car_details"):
    col1, col2 = st.columns(2)

    with col1:
        brand = st.selectbox("Brand", meta["brands"])
        fuel = st.selectbox("Fuel type", meta["fuel_types"])
        transmission = st.radio("Transmission", meta["transmissions"], horizontal=True)

    with col2:
        min_age, max_age = meta["car_age_range"]
        car_age = st.slider("Car age (years)", min_value=min_age, max_value=max_age,
                             value=min(5, max_age))
        km_driven = st.number_input(
            "Kilometers driven", min_value=0,
            max_value=int(meta["km_driven_range"][1] * 1.5),
            value=45000, step=1000,
        )
        seller_type = st.selectbox("Seller type", meta["seller_types"])

    owner = st.selectbox("Ownership history", meta["owner_types"])

    submitted = st.form_submit_button("Estimate price", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "brand": brand,
        "car_age": car_age,
        "km_driven": km_driven,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }])

    prediction = float(model.predict(row)[0])
    prediction = max(prediction, 0)

    st.subheader(format_inr(prediction))

    rmse = meta["test_rmse"]
    low, high = max(prediction - rmse, 0), prediction + rmse
    st.write(
        f"Typical range for this estimate: **{format_inr(low)} – {format_inr(high)}** "
        f"(based on the model's average error of about {format_inr(rmse)} on cars it "
        "hadn't seen during training)."
    )

    brand_median = meta["median_price_by_brand"].get(brand)
    if brand_median:
        diff_pct = (prediction - brand_median) / brand_median * 100
        direction = "above" if diff_pct >= 0 else "below"
        article = "an" if brand[0] in "AEIOU" else "a"
        st.write(
            f"This is **{abs(diff_pct):.0f}% {direction}** the median listing price "
            f"for {article} {brand} in the training data ({format_inr(brand_median)})."
        )

    st.caption(
        f"Model: {meta['chosen_model']} · trained on {meta['trained_rows']:,} listings "
        f"· R² on held-out data: {meta['test_r2']:.2f}"
    )

st.divider()
st.caption(
    "Built by Anaya Farhan as part of the AI/ML Internship (Task 04) at Devixo Solutions. "
    "Dataset: CarDekho used-car listings."
)
