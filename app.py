from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

# Expected feature columns after pd.get_dummies() during training
MODEL_COLUMNS = [
    "Id", "Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt",
    "Location_Downtown", "Location_Rural", "Location_Suburban", "Location_Urban",
    "Condition_Excellent", "Condition_Fair", "Condition_Good", "Condition_Poor",
    "Garage_No", "Garage_Yes"
]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    data = {
        "Id": float(request.form.get("Id", 0)),
        "Area": float(request.form["Area"]),
        "Bedrooms": float(request.form["Bedrooms"]),
        "Bathrooms": float(request.form["Bathrooms"]),
        "Floors": float(request.form["Floors"]),
        "YearBuilt": float(request.form["YearBuilt"]),
        "Location": request.form["Location"],
        "Condition": request.form["Condition"],
        "Garage": request.form["Garage"]
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)

    for col in MODEL_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    df = df[MODEL_COLUMNS]

    prediction = model.predict(df.values)

    return render_template("index.html", prediction_text=f"Predicted Price: {prediction[0]:.2f}")

if __name__ == "__main__":
    app.run(debug=True)