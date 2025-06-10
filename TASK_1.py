import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Downloads/Delinquency_prediction_dataset.csv")
df.info()
df.columns
df['Employment_Status'] = df['Employment_Status'].replace("EMP","Employed")
df['Employment_Status']
#removing the columns which are all containing null or missing values
removed_null_values_dataset = df.dropna(axis=1)
df.columns
removed_null_values_dataset.columns
numerics = [ 'int64', 'float64']
newdf = df.select_dtypes(include=numerics)
correlation_matrix = newdf.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
correlation_value = correlation_matrix['Delinquent_Account'].to_dict()
values = list(np.array(correlation_value.values()).max())
values[0:3]
H = df['Age'].fillna(df['Age'].mean(), inplace=True)
df["CreditUtilization"] = df.groupby("Credit_Score")["Credit_Utilization"].transform(lambda x: x.fillna(x.median()))
mean_income = df["Income"].mean()
std_income = df["Income"].std()
missing_count = df["Income"].isnull().sum()
synthetic_values = np.random.normal(mean_income, std_income, missing_count)
df.loc[df["Income"].isnull(), "Income"] = synthetic_values
sns.histplot(df["Debt_to_Income_Ratio"])
plt.show()
# Credit Utilization vs. Missed Payments
sns.boxplot(x=df["Missed_Payments"], y=df["Credit_Utilization"])
plt.title("Credit Utilization vs. MissedPayments")
plt.show()
# Debt-to-Income Ratio vs. Delinquent Account
sns.histplot(df[df["Delinquent_Account"] == 1]["Debt_to_Income_Ratio"], kde=True, color="red", label="Delinquent")
sns.histplot(df[df["Delinquent_Account"] == 0]["Debt_to_Income_Ratio"], kde=True, color="blue", label="Non-Delinquent")
plt.title("Debt-to-Income Ratio Distribution")
plt.legend()
plt.show()
