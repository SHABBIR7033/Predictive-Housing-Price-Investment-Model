new live_input = pd.DataFrame([[3000, 4, 2, 1]], columns=X.columns)

predicted_value = model.predict(live_input)
print(f"Predicted Price for the house: ₹{predicted_value[0]:,.2f}")

