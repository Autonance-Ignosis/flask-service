import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall

# Load dataset
df = pd.read_csv('/Users/krispatel/Desktop/Autonance/flask-service/datasets/loan_data.csv')

# Define columns
numerical_cols = [
    'interest_rate', 'installment', 'log_of_income', 'debt_income_ratio',
    'fico_score', 'days_with_credit_line', 'revolving_balance',
    'revolving_utilization'
]
categorical_cols = [
    'credit_criteria_meet', 'purpose', 'inquiry_last_6months',
    'times_surpassed_payment_in_2yrs', 'derogatory_public_record'
]
target_col = 'loan_defaulted'

# Standard scale numerical features
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[numerical_cols]), columns=numerical_cols)
joblib.dump(scaler, 'scaler.pkl')

# Combine scaled numericals with categorical and target
df_combined = pd.concat([df_scaled, df[categorical_cols + [target_col]]], axis=1)

# One-hot encode categorical features (only 'purpose' here)
df_encoded = pd.get_dummies(df_combined, columns=['purpose'])
model_columns = df_encoded.drop(target_col, axis=1).columns.tolist()
joblib.dump(model_columns, 'model_columns.pkl')

# Features and target
X = df_encoded.drop(target_col, axis=1)
Y = df_encoded[target_col]

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_balanced, Y_balanced = smote.fit_resample(X, Y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X_balanced, Y_balanced, test_size=0.2, random_state=42
)

# Convert to float32 for TensorFlow
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Build model
model = Sequential([
    Input(shape=(x_train.shape[1],)),
    Dense(25, activation='relu'),
    BatchNormalization(),
    Dense(55, activation='relu'),
    BatchNormalization(),
    Dense(35, activation='relu'),
    BatchNormalization(),
    Dense(25, activation='relu'),
    BatchNormalization(),
    Dense(1, activation='sigmoid')
])

# Compile
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)

# Train
model.fit(
    x_train, y_train,
    epochs=120,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=2
)

# Save model in modern format
model.save('Loan_default.keras', save_format='keras')
