import pandas as pd
import joblib

# Load dataset
data = pd.read_csv("House Price Prediction Dataset.csv")

# Check columns
print(data.columns)

# Separate features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Convert categorical columns to numeric
X = pd.get_dummies(X)

# Train model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")
