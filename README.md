# Telco Customer Churn Prediction

A complete machine learning project to predict telecom customer churn using Logistic Regression and Naive Bayes models.

**Live App**: Coming soon after deployment!

## Project Goal
Predict which customers are likely to churn so the company can offer targeted retention strategies (discounts, calls, better plans) and save revenue.

## Key Insights from EDA
- Month-to-month contract customers have the highest churn
- New customers (low tenure) churn quickly
- Fiber optic internet users churn more than DSL/No internet
- Electronic check payment users have higher churn
- High monthly charges increase churn risk
- Customers with add-on services (security, tech support, streaming) are more loyal

## Models
- **Logistic Regression** (Recommended – realistic probabilities)
- **Naive Bayes** (High recall – catches maximum potential churn)

App mein sidebar se model switch kar sakte ho.

## Features
- Interactive Streamlit web app
- Auto-calculated Total Charges
- Real-time churn prediction with probability
- Business suggestions for high-risk customers

## How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/Azam-Khan12/Telco-Churn-Prediction.git
   cd Telco-Churn-Prediction
