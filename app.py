import streamlit as st
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Create Dataset
# -----------------------------
np.random.seed(42)
n_samples = 1000

sqft = np.random.randint(800, 4000, n_samples)
age = np.random.randint(1, 25, n_samples)
city = np.random.choice(['Delhi', 'Noida', 'Gurugram'], n_samples)

price = (sqft * 4500) - (age * 30000) + np.random.normal(0, 150000, n_samples)

for i in range(n_samples):
    if city[i] == "Delhi":
        price[i] += 600000
    elif city[i] == "Gurugram":
        price[i] += 400000

df = pd.DataFrame({
    "SquareFeet": sqft,
    "Age": age,
    "City": city,
    "Price": price
})

df_encoded = pd.get_dummies(df, columns=["City"], drop_first=True)

X = df_encoded.drop("Price", axis=1)
y = df_encoded["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="centered"
)

st.title("🏡 House Price Prediction")

st.write("Predict house prices using Machine Learning.")

square_feet = st.number_input(
    "Square Feet",
    min_value=800,
    max_value=4000,
    value=2000
)

house_age = st.slider(
    "House Age (Years)",
    1,
    25,
    5
)

city_option = st.selectbox(
    "Select City",
    ["Delhi", "Noida", "Gurugram"]
)

# One-Hot Encoding
city_gurugram = 1 if city_option == "Gurugram" else 0
city_noida = 1 if city_option == "Noida" else 0

input_df = pd.DataFrame(
    [[square_feet, house_age, city_gurugram, city_noida]],
    columns=X.columns
)

if st.button("Predict Price"):

    prediction = model.predict(input_df)[0]

    st.success(
        f"Estimated House Price: ₹{prediction:,.2f}"
    )
