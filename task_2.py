import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset (1).xlsx")
# Handle missing values (Example: Loan_Balance with mean)
df["Loan_Balance"].fillna(df["Loan_Balance"].mean(), inplace=True)
# Encode categorical variables
categorical_cols = ["Employment_Status", "Credit_Card_Type", "Location"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# Select relevant features
features = ["Credit_Utilization", "Missed_Payments", "Debt_to_Income_Ratio", "Loan_Balance", "Credit_Score"]
X = df[features]
y = df["Delinquent_Account"]
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize numerical values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)
# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
# Evaluate models
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_lr):.2f}")
print("\nRandom Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_rf):.2f}")
