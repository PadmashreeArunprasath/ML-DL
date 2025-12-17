import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# STEP 2: CHECK FILE LOCATION
# -----------------------------
print("Current Working Directory:")
print(os.getcwd())

# Check if files exist
if not os.path.exists("train.csv") or not os.path.exists("test.csv"):
    raise FileNotFoundError(
   
     
    )

# -----------------------------
# STEP 3: LOAD DATASET
# -----------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("\nTrain Data Loaded Successfully!")
print(train.head())

# -----------------------------
# STEP 4: DATA CLEANING
# -----------------------------

# Fill missing Age with median
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# Fill missing Embarked with mode
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])

# Fill missing Fare in test set
test["Fare"] = test["Fare"].fillna(test["Fare"].median())

# -----------------------------
# STEP 5: ENCODE CATEGORICAL DATA
# -----------------------------
le = LabelEncoder()

train["Sex"] = le.fit_transform(train["Sex"])
test["Sex"] = le.transform(test["Sex"])

train["Embarked"] = le.fit_transform(train["Embarked"])
test["Embarked"] = le.transform(test["Embarked"])

# -----------------------------
# STEP 6: FEATURE SELECTION
# -----------------------------
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

X = train[features]
y = train["Survived"]

X_test_final = test[features]

# -----------------------------
# STEP 7: TRAIN-TEST SPLIT
# -----------------------------
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 8: LOGISTIC REGRESSION
# -----------------------------
log_model = LogisticRegression(max_iter=300)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_valid)
log_acc = accuracy_score(y_valid, y_pred_log)

print("\n🔹 Logistic Regression Accuracy:", log_acc)

# -----------------------------
# STEP 9: RANDOM FOREST MODEL
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=7,
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_valid)
rf_acc = accuracy_score(y_valid, y_pred_rf)

print("🔹 Random Forest Accuracy:", rf_acc)

# -----------------------------
# STEP 10: MODEL EVALUATION
# -----------------------------
print("\n📊 Classification Report (Random Forest):")
print(classification_report(y_valid, y_pred_rf))

plt.figure(figsize=(6,4))
sns.heatmap(
    confusion_matrix(y_valid, y_pred_rf),
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -----------------------------
# STEP 11: PREDICTION ON TEST DATA
# -----------------------------
test_predictions = rf_model.predict(X_test_final)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_predictions
})

submission.to_csv("titanic_submission.csv", index=False)

print("\n✅ SUCCESS!")
print("📁 File created: titanic_submission.csv")