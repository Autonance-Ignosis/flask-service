from flask import Blueprint, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


model = keras.models.load_model('/Users/krispatel/Desktop/Autonance/flask-service/models/Loan_default.h5')

loan_bp = Blueprint('loan_bp', __name__)

@loan_bp.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()
    
    # Create a DataFrame for the incoming data
    X_sample = pd.DataFrame(data)
    
    # Initialize the scaler (ensure to use the same  as during training)
    scaler = StandardScaler()
    
    # Define the numerical columns to be scaled
    numerical_columns = [
        'interest_rate', 'installment', 'log_of_income', 'debt_income_ratio',
        'fico_score', 'days_with_credit_line', 'revolving_balance',
        'revolving_utilization', 'credit_criteria_meet', 'inquiry_last_6months',
        'times_surpassed_payment_in_2yrs', 'derogatory_public_record'
    ]
    
    # Scale numerical columns
    X_sample[numerical_columns] = scaler.fit_transform(X_sample[numerical_columns])
    
    # Perform prediction using the model
    prediction = model.predict(X_sample)
    
    # Return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})
