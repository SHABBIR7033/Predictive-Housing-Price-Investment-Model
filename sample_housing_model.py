import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# STEP 1: CREATE THE DATASET (Pandas)

print("--- Step 1: Preparing Data ---")
np.random.seed(42)
n_samples = 1000

# Generating raw features
sqft = np.random.randint(800, 4000, n_samples)
age = np.random.randint(1, 25, n_samples)
city = np.random.choice(['Delhi', 'Noida', 'Gurugram'], n_samples)

# Formula to calculate realistic prices with some random market variation
price = (sqft * 4500) - (age * 30000) + np.random.normal(0, 150000, n_samples)
for i in range(n_samples):
    if city[i] == 'Delhi': 
        price[i] += 600000
    elif city[i] == 'Gurugram': 
        price[i] += 400000

# Combine everything into a single DataFrame
df = pd.DataFrame({'SquareFeet': sqft, 'Age': age, 'City': city, 'Price': price})
print(df.head()) # Shows the first 5 rows


# STEP 2: CONVERT TEXT TO NUMBERS (One-Hot Encoding)
print("\n--- Step 2: Converting City text to numerical flags ---")
# Machine learning models only understand numbers. This creates binary columns for cities.
df_encoded = pd.get_dummies(df, columns=['City'], drop_first=True)


# STEP 3: SPLIT DATA INTO TRAIN & TEST SETS

print("\n--- Step 3: Splitting Dataset ---")
# X = Input variables (Features), y = Target variable (What we want to predict)
X = df_encoded.drop('Price', axis=1)
y = df_encoded['Price']

# 80% data to train the model, 20% to test its accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")


# STEP 4: TRAIN THE MACHINE LEARNING MODEL

print("\n--- Step 4: Training Random Forest Regressor ---")
# Initializing the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model on our data
model.fit(X_train, y_train)
print("Model training completed successfully!")


# STEP 5: EVALUATE PERFORMANCE

print("\n--- Step 5: Calculating Accuracy ---")
# Predict prices for the unseen test data
y_pred = model.predict(X_test)

# Calculate R2 Score (Accuracy measure from 0 to 1)
accuracy = r2_score(y_test, y_pred)
print(f"Model R² Score (Accuracy): {accuracy:.2f}")


# STEP 6: MAKE A LIVE PREDICTION

print("\n--- Step 6: Testing with New Live Inputs ---")
# Example: Predict price for a 2000 SqFt house, 5 years old, located in Gurugram
# Columns expected by the model: ['SquareFeet', 'Age', 'City_Gurugram', 'City_Noida']
live_input = pd.DataFrame([[3000, 4, 2, 1]], columns=X.columns)

predicted_value = model.predict(live_input)
print(f"Predicted Price for the house: ₹{predicted_value[0]:,.2f}")
