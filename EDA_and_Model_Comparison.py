# Cell 1: Imports & Data Load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import *

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Cell 2: Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['TotalCharges', 'Churn'])
df.drop('customerID', axis=1, inplace=True)

# Cell 3: EDA Graphs
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
sns.countplot(x='Churn', data=df)
plt.title('Churn Count')

plt.subplot(2, 3, 2)
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title('Churn by Contract')

df['tenure_group'] = pd.cut(df['tenure'], bins=[0,12,24,36,48,60,72], labels=['0-12','13-24','25-36','37-48','49-60','61-72'])
plt.subplot(2, 3, 3)
sns.countplot(x='tenure_group', hue='Churn', data=df)
plt.title('Churn by Tenure Group')

plt.subplot(2, 3, 4)
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn by Internet Service')

plt.subplot(2, 3, 5)
sns.countplot(x='PaymentMethod', hue='Churn', data=df)
plt.xticks(rotation=45)
plt.title('Churn by Payment Method')

plt.subplot(2, 3, 6)
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges by Churn')

plt.tight_layout()
plt.show()

df.drop('tenure_group', axis=1, inplace=True)

# Cell 4: Preprocessing
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cell 5: Model Comparison
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'LDA': LinearDiscriminantAnalysis()
}

results = []
for name, model in models.items():
    if name in ['Logistic Regression', 'KNN', 'LDA']:
        model.fit(X_train_scaled, y_train)
        pred = model.predict(X_test_scaled)
        prob = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, "predict_proba") else None
    else:
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    
    results.append({
        'Model': name,
        'Accuracy': round(accuracy_score(y_test, pred), 4),
        'Precision': round(precision_score(y_test, pred), 4),
        'Recall': round(recall_score(y_test, pred), 4),
        'F1': round(f1_score(y_test, pred), 4),
        'AUC': round(roc_auc_score(y_test, prob), 4) if prob is not None else 'N/A',
        'Missed Churn': confusion_matrix(y_test, pred)[1,0]
    })

print(pd.DataFrame(results).to_string(index=False))