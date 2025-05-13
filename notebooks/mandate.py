import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("/Users/krispatel/Desktop/Autonance/flask-service/datasets/mandate_dataset.csv")

# Show basic info
print(df.info())

# Check for null/missing values
print("\nMissing Values:\n", df.isnull().sum())

# Distribution of target variable
print("\nStatus Distribution:\n", df['status'].value_counts())


# Encode categorical columns
cat_cols = ['category', 'mandate_variant', 'debit_type', 'seq_type', 'freq_type']
le_dict = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Ensure all are strings
    le_dict[col] = le  # Save the encoder

# Convert date columns to durations
df['start_date'] = pd.to_datetime(df['start_date'], errors='coerce')
df['upto_date'] = pd.to_datetime(df['upto_date'], errors='coerce')
df['duration_days'] = (df['upto_date'] - df['start_date']).dt.days
df.drop(columns=['start_date', 'upto_date'], inplace=True)

# Encode boolean
df['up_to_40_years'] = df['up_to_40_years'].astype(int)

# Encode target column
df['status'] = df['status'].map({'APPROVED': 1, 'REJECTED': 0})

# Handle missing values if any
df.dropna(inplace=True)

# Scale amount
scaler = StandardScaler()
df['amount_scaled'] = scaler.fit_transform(df[['amount']])
df.drop(columns=['amount'], inplace=True)

# Split into features and target
X = df.drop(columns=['status'])
y = df['status']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train the model
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and preprocessing tools
joblib.dump(model, "mandate_classifier.pkl")
joblib.dump(scaler, "amount_scaler.pkl")
joblib.dump(le_dict, "label_encoders.pkl")
