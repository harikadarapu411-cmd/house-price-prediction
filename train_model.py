import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load dataset
data = pd.read_csv("House Price Prediction Dataset.csv")

# -----------------------------
# 1. Drop unnecessary column
# -----------------------------
# Id is not useful for prediction
if "Id" in data.columns:
    data = data.drop("Id", axis=1)

# -----------------------------
# 2. Separate features & target
# -----------------------------
X = data.drop("Price", axis=1)
y = data["Price"]

# -----------------------------
# 3. Convert categorical data
# -----------------------------
X = pd.get_dummies(X)

# Save column names for later use in Flask
model_columns = X.columns.tolist()
joblib.dump(model_columns, "model_columns.pkl")

# -----------------------------
# 4. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 5. Train model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# 6. Save model
# -----------------------------
joblib.dump(model, "model.pkl")

print("Model trained and saved successfully!")

