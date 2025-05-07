from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from flask import Blueprint

mandate_bp = Blueprint('mandate', __name__)


model = joblib.load("/Users/krispatel/Desktop/Autonance/flask-service/models/mandate_classifier.pkl")
scaler = joblib.load("/Users/krispatel/Desktop/Autonance/flask-service/models/amount_scaler.pkl")
le_dict = joblib.load("/Users/krispatel/Desktop/Autonance/flask-service/models/label_encoders.pkl")


@mandate_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical fields
        for col in le_dict:
            le = le_dict[col]
            df[col] = le.transform([df[col].iloc[0]])

        # Convert boolean
        df['up_to_40_years'] = df['up_to_40_years'].astype(int)


        # Compute duration from start_date and upto_date
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['upto_date'] = pd.to_datetime(df['upto_date'])
        df['duration_days'] = (df['upto_date'] - df['start_date']).dt.days
        df.drop(columns=['start_date', 'upto_date'], inplace=True)

        # Scale amount
        df['amount_scaled'] = scaler.transform([[df['amount'].iloc[0]]])
        df.drop(columns=['amount'], inplace=True)

        # Make prediction
        proba = model.predict_proba(df)[0]  # [prob_rejected, prob_approved]
        prediction = int(proba[1] >= 0.5)
        status = "APPROVED" if prediction == 1 else "REJECTED"

        return jsonify({
            "prediction": prediction,
            "status": status,
            "probability": {
                "APPROVED": round(float(proba[1]), 4),
                "REJECTED": round(float(proba[0]), 4)
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400
