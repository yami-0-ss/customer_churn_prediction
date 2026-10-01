
📉 Customer Churn Prediction

A Machine Learning-based Customer Churn Prediction System that predicts whether a customer is likely to stay or leave a service based on customer and account-related information.
live project link = https://customer-churn-prediction-1-8biq.onrender.com/

📌 Project Overview

Customer churn is an important business problem where companies lose customers over time.

This project uses Machine Learning classification techniques to predict customer churn based on relevant customer attributes.

The system provides a simple web interface where users can enter customer details and receive a churn prediction.

🎯 Objective

The main objective is to help businesses:

Identify customers who may leave
Understand customer churn patterns
Take early retention actions
Improve customer retention
Support data-driven decision making
🚀 Features
📊 Customer churn prediction
🤖 Machine Learning classification
🧹 Data preprocessing
🔍 Exploratory Data Analysis
📈 Feature engineering
🖥️ User-friendly prediction interface
🌐 Deployed web application
⚡ Real-time prediction
🧠 Machine Learning Workflow
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Train-Test Split
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization
   ↓
Flask Web Application
   ↓
Render Deployment
📊 Input Features

The model can use customer-related information such as:

Customer demographics
Account information
Service details
Contract information
Payment-related information
Customer tenure
Monthly/total charges
Other relevant customer attributes
🎯 Prediction

The application predicts the customer's churn status.

Customer Details
       ↓
Machine Learning Model
       ↓
Churn Prediction
       ↓
Yes / No

Example:

Prediction: Customer is likely to Churn

or

Prediction: Customer is likely to Stay
🛠️ Technologies Used
Technology	Purpose
Python	Programming
Pandas	Data Processing
NumPy	Numerical Operations
Scikit-learn	Machine Learning
Matplotlib	Data Visualization
Seaborn	Exploratory Data Analysis
Flask	Web Application
HTML/CSS	Frontend
Pickle	Model Serialization
Render	Deployment
GitHub	Version Control
🤖 Machine Learning

This project follows a supervised learning approach.

Steps

1. Data Preprocessing

Handle missing values
Remove unnecessary columns
Encode categorical variables
Prepare numerical features

2. Exploratory Data Analysis

Analyze customer characteristics
Identify churn patterns
Visualize important features

3. Model Training

Classification algorithms can be used to learn the relationship between customer attributes and churn.

4. Model Evaluation

The trained model can be evaluated using metrics such as:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
📂 Project Structure
Customer-Churn-Prediction/
│
├── app.py
├── model.pkl
├── requirements.txt
├── README.md
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
└── dataset/
    └── customer_churn.csv
▶️ How to Run Locally
1. Clone the Repository
git clone https://github.com/your-username/Customer-Churn-Prediction.git
2. Navigate to the Project
cd Customer-Churn-Prediction
3. Create Virtual Environment
python -m venv venv
4. Activate Environment

Windows:

venv\Scripts\activate
5. Install Dependencies
pip install -r requirements.txt
6. Run the Application
python app.py
7. Open in Browser
http://127.0.0.1:5000/
📦 Requirements

Example:

Flask
pandas
numpy
scikit-learn
matplotlib
seaborn
gunicorn

📈 Business Use Case

Customer churn prediction can help organizations identify customers who may leave and take appropriate retention actions.

For example:

High Churn Risk
       ↓
Identify Customer
       ↓
Analyze Customer Behavior
       ↓
Offer Retention Strategy
       ↓
Improve Customer Retention
🔮 Future Enhancements
Add customer churn probability
Add interactive analytics dashboard
Add customer segmentation
Add feature importance visualization
Compare multiple ML models
Add prediction history
Add database integration
Add customer retention recommendations

👩‍💻 Author

Yamini More

Aspiring Data Analyst / Data Scientist

Skills Demonstrated

Python • Machine Learning • Pandas • NumPy • Scikit-learn • Flask • Data Analysis • Data Visualization
