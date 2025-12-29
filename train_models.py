import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import pickle
import os

# Data load
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['TotalCharges', 'Churn'])
df.drop('customerID', axis=1, inplace=True)

# Encoding
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

os.makedirs('models', exist_ok=True)

# ==================== LOGISTIC REGRESSION ====================
print("Training Logistic Regression...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train_scaled, y_train)

with open('models/churn_model_logistic.pkl', 'wb') as f:
    pickle.dump(model_logistic, f)

with open('models/scaler_logistic.pkl', 'wb') as f:
    pickle.dump(scaler, f)

preprocess_info = {'categorical_cols': categorical_cols, 'feature_names': X.columns.tolist()}
with open('models/preprocess_info_logistic.pkl', 'wb') as f:
    pickle.dump(preprocess_info, f)

print("Logistic Regression saved!\n")

# ==================== NAIVE BAYES ====================
print("Training Naive Bayes...")
model_naive = GaussianNB()
model_naive.fit(X_train, y_train)  # No scaling

with open('models/churn_model_naive.pkl', 'wb') as f:
    pickle.dump(model_naive, f)

with open('models/preprocess_info_naive.pkl', 'wb') as f:
    pickle.dump(preprocess_info, f)  # Same info, no scaler

print("Naive Bayes saved!")
print("\nBoth models trained and saved in models/ folder!")

# Ye lines file ke end mein add kar (print ke liye)

print("\nCleaned Data Head:")
print(df.head())

print("\nEncoded Data Head (final X):")
print(X.head())

print("\nFinal Features List:")
print(X.columns.tolist())

print("\nShape after encoding:", X.shape)

df_encoded.to_excel('models/processed_data.xlsx', index=False)
print("Processed data saved as Excel: models/processed_data.xlsx")