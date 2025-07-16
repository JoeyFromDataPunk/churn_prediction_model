Customer Churn Prediction App

Overview: This Streamlit web app predicts customer churn using machine learning models based on telecom data.

Users can explore data visualizations, compare different K-Nearest Neighbors (KNN) models, and make predictions by entering customer-specific information.

Features
Interactive Visualizations:
- Parallel Coordinates Plot
- Pair Plots
- Correlation Heatmaps

Model Comparison
Four KNN models are tested and compared using confusion matrices, crosstabs, and classification reports.

Cross-Validation and K-Tuning
Each model is tuned using 10-fold cross-validation to select the optimal k value.

User Prediction Tool
Enter customer service calls and day charges to predict whether a customer will churn.

How to Run Locally
1.) Install Requirements
pip install -r requirements.txt

2.) Run the App
streamlit run Churn_Model.py

Deployment
This app is deployed on Streamlit Cloud:
View the App Here
(Replace with your actual Streamlit Cloud link)

Input Variables for Prediction
Features:
Customer Service Calls
  - Number of times the customer called support
Total Day Charge
  - Daytime charges in dollars

Dataset
Churn.csv
Telecom dataset with 14 features, including usage patterns, charges, and customer service interactions.

Author
Joseph Boyle
University of Maryland Global Campus – Data Science Program

