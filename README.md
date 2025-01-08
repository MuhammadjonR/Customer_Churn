# Customer Churn Prediction Model
Overview
This repository contains a Machine Learning model designed to predict customer churn. The model leverages features such as tenure, complaints, cashback amount, marital status, and gender to forecast whether a customer is likely to churn.

Table of Contents
Overview
Features
Installation
Usage
Evaluation
License
Features
Predict customer churn.
Handles both categorical and numerical data.
Built using XGBoost for improved accuracy.
Installation
Clone the repository:

git clone https://github.com/yourusername/ML_CustomerChurn.git
Navigate to the project directory:


cd ML_CustomerChurn
Install the necessary dependencies:



pip install -r requirements.txt

Usage
Training the Model: Train the model using historical customer data by following the steps in training.py.

Making Predictions: You can use the Streamlit application to upload a CSV file containing customer data and predict churn.


streamlit run customer_churn.py
Model Deployment: The trained model is saved as xgb_model.pkl and can be used for batch predictions or real-time predictions.

Evaluation
Model Performance: After training, evaluate the model using various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

y_pred = xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Model accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt="g")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
plt.show()
![image](https://github.com/user-attachments/assets/d19bf540-85c3-4b38-a4d5-a0d1ca85cd24)

