from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained model and saved feature columns
model = joblib.load("model.pkl")
MODEL_COLUMNS = joblib.load("model_columns.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        data = {
            "Area": float(request.form["Area"]),
            "Bedrooms": float(request.form["Bedrooms"]),
            "Bathrooms": float(request.form["Bathrooms"]),
            "Floors": float(request.form["Floors"]),
            "YearBuilt": float(request.form["YearBuilt"]),
            "Location": request.form["Location"],
            "Condition": request.form["Condition"],
            "Garage": request.form["Garage"]
        }

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Apply one-hot encoding
        df = pd.get_dummies(df)

        # Add missing columns
        for col in MODEL_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        # Ensure correct order
        df = df[MODEL_COLUMNS]

        # Prediction
        prediction = model.predict(df)

        return render_template(
            "index.html",
            prediction_text=f"Predicted Price: {prediction[0]:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

# For Render.com deployment
port = int(os.environ.get("PORT", 5000))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=port, debug=False)