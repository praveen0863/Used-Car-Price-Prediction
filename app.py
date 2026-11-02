
from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load trained model and preprocessor
model = joblib.load("models/best_car_price_model.pkl")
preprocessor = joblib.load("models/car_preprocessor.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        car_data = {
            "Year": int(request.form["year"]),
            "Mileage_km": float(request.form["mileage"]),
            "Fuel_Type": request.form["fuel"],
            "Transmission": request.form["transmission"],
            "Brand": request.form["brand"],
            "Vehicle_Type": request.form["vehicle_type"],
            "Previous_Owners": int(request.form["owners"]),
            "Engine_CC": int(request.form["engine"])
        }

        input_df = pd.DataFrame([car_data])

        # Apply the saved preprocessing
        processed_data = preprocessor.transform(input_df)

        # Predict car price
        prediction = model.predict(processed_data)[0]

        return render_template(
            "index.html",
            prediction_text=f"Estimated Car Price: ₹{prediction:.2f} Lakh"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
