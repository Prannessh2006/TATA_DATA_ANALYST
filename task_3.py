# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# Load dataset
df = pd.read_excel("Delinquency_prediction_dataset.xlsx")

# Handle missing values (Example: Loan_Balance with mean imputation)
df["Loan_Balance"].fillna(df["Loan_Balance"].mean(), inplace=True)

# Encode categorical variables
categorical_cols = ["Employment_Status", "Credit_Card_Type", "Location"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Define features and target variable
features = ["Credit_Utilization", "Missed_Payments", "Debt_to_Income_Ratio", "Loan_Balance", "Credit_Score"]
X = df[features]
y = df["Delinquent_Account"]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical values
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
lr_model = LogisticRegression()
rf_model = RandomForestClassifier()
lr_model.fit(X_train_scaled, y_train)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test_scaled)
y_pred_rf = rf_model.predict(X_test)

# Evaluate models
print("Logistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_lr):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))
print("\nRandom Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf):.2f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_rf):.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

# Fairness audit: Check risk prediction across different customer segments
fairness_groups = ["Age", "Credit_Score"]
for group in fairness_groups:
    avg_risk = df.groupby(group)["Delinquent_Account"].mean()
    print(f"\nAverage delinquency risk by {group}:")
    print(avg_risk)

# Data Visualization
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Between Variables")
plt.show()
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["Missed_Payments"], y=df["Credit_Utilization"])
plt.title("Credit Utilization vs. Missed Payments")
plt.show()
plt.figure(figsize=(8, 5))
sns.histplot(df[df["Delinquent_Account"] == 1]["Debt_to_Income_Ratio"], kde=True, color="red", label="Delinquent")
sns.histplot(df[df["Delinquent_Account"] == 0]["Debt_to_Income_Ratio"], kde=True, color="blue", label="Non-Delinquent")
plt.title("Debt-to-Income Ratio Distribution")
plt.legend()
plt.show()
