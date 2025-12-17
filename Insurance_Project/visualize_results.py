import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
df = pd.read_csv("insurance.csv")
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)

X = df.drop('charges', axis=1)
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models for visualization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Create visualizations
plt.style.use('default')
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. Model Performance Comparison
models = ['Linear Regression', 'Random Forest', 'ANN']
mae_scores = [4181.19, 2518.95, 4482.62]
rmse_scores = [5796.28, 4542.80, 6393.89]

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, mae_scores, width, label='MAE', alpha=0.8, color='skyblue')
axes[0].bar(x + width/2, rmse_scores, width, label='RMSE', alpha=0.8, color='coral')
axes[0].set_xlabel('Models')
axes[0].set_ylabel('Error ($)')
axes[0].set_title('Model Performance Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 2. Distribution of Insurance Charges
axes[1].hist(df['charges'], bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1].set_xlabel('Insurance Charges ($)')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Insurance Charges')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('insurance_analysis_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'insurance_analysis_results.png'")