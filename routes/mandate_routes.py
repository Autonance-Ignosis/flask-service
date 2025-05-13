from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import numpy as np

mandate_bp = Blueprint('mandate', __name__)

model = joblib.load("/Users/krispatel/Desktop/Autonance/flask-service/models/mandate/mandate_classifier.pkl")
scaler = joblib.load("/Users/krispatel/Desktop/Autonance/flask-service/models/mandate/amount_scaler.pkl")
le_dict = joblib.load("/Users/krispatel/Desktop/Autonance/flask-service/models/mandate/label_encoders.pkl")

@mandate_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure required fields are present
        required_fields = ['category', 'mandate_variant', 'debit_type', 'seq_type', 'freq_type', 
                           'up_to_40_years', 'start_date', 'upto_date', 'amount']
        for field in required_fields:
            if field not in df.columns:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Label encode categorical fields
        for col in le_dict:
            if col in df.columns:
                le = le_dict[col]
                df[col] = le.transform([df[col].iloc[0]])
            else:
                return jsonify({"error": f"Missing categorical column: {col}"}), 400

        # Boolean conversion
        df['up_to_40_years'] = df['up_to_40_years'].astype(int)

        # Duration calculation
        df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
        df['upto_date'] = pd.to_datetime(df['upto_date'], errors='coerce')
        if df['start_date'].isnull().any() or df['upto_date'].isnull().any():
            return jsonify({"error": "Invalid date format"}), 400
        df['duration_days'] = (df['upto_date'] - df['start_date']).dt.days
        df.drop(columns=['start_date', 'upto_date'], inplace=True)

        # Scale amount
        df['amount_scaled'] = scaler.transform([[df['amount'].iloc[0]]])
        df.drop(columns=['amount'], inplace=True)

        # Predict
        proba = model.predict_proba(df)[0]
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
        return jsonify({"error": f"Exception occurred: {str(e)}"}), 500
