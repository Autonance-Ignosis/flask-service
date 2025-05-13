from flask import Blueprint, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Load the model and preprocessing artifacts
model = tf.keras.models.load_model('/Users/krispatel/Desktop/Autonance/flask-service/models/loan/Loan_default.keras')
model_columns = joblib.load('/Users/krispatel/Desktop/Autonance/flask-service/models/loan/model_columns.pkl')
scaler = joblib.load('/Users/krispatel/Desktop/Autonance/flask-service/models/loan/scaler.pkl')

loan_bp = Blueprint('loan_bp', __name__)

@loan_bp.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Convert input to DataFrame
        df_input = pd.DataFrame([data])

        # Define numerical columns
        numerical_cols = [
            'interest_rate', 'installment', 'log_of_income', 'debt_income_ratio',
            'fico_score', 'days_with_credit_line', 'revolving_balance',
            'revolving_utilization'
        ]

        # Ensure all required numerical columns are present
        missing_numericals = [col for col in numerical_cols if col not in df_input.columns]
        if missing_numericals:
            return jsonify({"error": f"Missing numerical fields: {missing_numericals}"}), 400

        # Scale numerical features
        df_input[numerical_cols] = scaler.transform(df_input[numerical_cols])

        # One-hot encode categorical column: 'purpose'
        df_input = pd.get_dummies(df_input)

        # Add missing columns
        for col in model_columns:
            if col not in df_input.columns:
                df_input[col] = 0

        # Ensure correct column order
        df_input = df_input[model_columns]

        # Predict
        prediction = model.predict(df_input.astype('float32'))[0][0]
        result = {
            'loan_default_probability': round(float(prediction), 4)
            # Optional: 'loan_defaulted': int(prediction >= 0.5)
        }

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
