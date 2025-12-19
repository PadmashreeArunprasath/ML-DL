import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Load dataset
df = pd.read_csv("insurance.csv")

print("Dataset Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nInsurance Charges (first 10 records):")
print(df['charges'].head(10).tolist())

# Preprocessing
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

print(f"\nFeatures after preprocessing: {X.columns.tolist()}")
print(f"Feature matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
print("\n" + "="*50)
print("LINEAR REGRESSION")
print("="*50)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

mae_lr = mean_absolute_error(y_test, y_pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
r2_lr = r2_score(y_test, y_pred_lr)

print(f"MAE: ${mae_lr:,.2f}")
print(f"RMSE: ${rmse_lr:,.2f}")
print(f"R² Score: {r2_lr:.4f}")

# Random Forest
print("\n" + "="*50)
print("RANDOM FOREST")
print("="*50)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

print(f"MAE: ${mae_rf:,.2f}")
print(f"RMSE: ${rmse_rf:,.2f}")
print(f"R² Score: {r2_rf:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\nTop 5 Most Important Features:")
for i, row in feature_importance.head().iterrows():
    print(f"{row['feature']}: {row['importance']:.4f}")

# ANN (Artificial Neural Network)
print("\n" + "="*50)
print("ARTIFICIAL NEURAL NETWORK")
print("="*50)

ann = Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])

ann.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training ANN...")
history = ann.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    verbose=0,
    callbacks=[early_stopping]
)
print("ANN training completed!")

y_pred_ann = ann.predict(X_test_scaled, verbose=0).flatten()

mae_ann = mean_absolute_error(y_test, y_pred_ann)
rmse_ann = np.sqrt(mean_squared_error(y_test, y_pred_ann))
r2_ann = r2_score(y_test, y_pred_ann)

print(f"MAE: ${mae_ann:,.2f}")
print(f"RMSE: ${rmse_ann:,.2f}")
print(f"R² Score: {r2_ann:.4f}")

# Model Comparison
print("\n" + "="*50)
print("MODEL COMPARISON SUMMARY")
print("="*50)

results = pd.DataFrame({
    'Model': ['Linear Regression', 'Random Forest', 'ANN'],
    'MAE': [mae_lr, mae_rf, mae_ann],
    'RMSE': [rmse_lr, rmse_rf, rmse_ann],
    'R² Score': [r2_lr, r2_rf, r2_ann]
})

results = results.sort_values('RMSE')
print(results.to_string(index=False, float_format='%.2f'))

print(f"\nBest Model: {results.iloc[0]['Model']} with RMSE of ${results.iloc[0]['RMSE']:,.2f}")

# Demonstrate predictions for new customers
print("\n" + "="*50)
print("SAMPLE PREDICTIONS FOR NEW CUSTOMERS")
print("="*50)

# Example customers (age, sex, bmi, children, smoker, region_northwest, region_southeast, region_southwest)
sample_customers = [
    [25, 0, 25.0, 0, 0, 1, 0, 0],  # Young male, normal BMI, non-smoker, northwest
    [45, 1, 30.0, 2, 1, 0, 1, 0],  # Middle-aged female, overweight, smoker, southeast
    [35, 0, 22.0, 1, 0, 0, 0, 1],  # Male, healthy BMI, non-smoker, southwest
]

customer_descriptions = [
    "25yr old male, BMI 25, no kids, non-smoker, northwest",
    "45yr old female, BMI 30, 2 kids, smoker, southeast", 
    "35yr old male, BMI 22, 1 kid, non-smoker, southwest"
]

# Scale the sample data
sample_scaled = scaler.transform(sample_customers)

# Predict using the best model (Random Forest)
predictions = rf.predict(sample_customers)

for i, (desc, pred) in enumerate(zip(customer_descriptions, predictions)):
    print(f"Customer {i+1}: {desc}")
    print(f"Predicted Insurance Charge: ${pred:,.2f}")
    print()

# Save the trained models
print("\n" + "="*50)
print("SAVING TRAINED MODELS")
print("="*50)

# Save Random Forest (best model) as .pkl
with open('best_model_random_forest.pkl', 'wb') as f:
    pickle.dump(rf, f)
print("✅ Random Forest model saved as 'best_model_random_forest.pkl'")

# Save Linear Regression as .pkl
with open('linear_regression_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("✅ Linear Regression model saved as 'linear_regression_model.pkl'")

# Save ANN as .h5
ann.save('ann_model.h5')
print("✅ ANN model saved as 'ann_model.h5'")

# Save scaler for future predictions
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved as 'scaler.pkl'")

print("\n🎉 All models saved successfully!")
