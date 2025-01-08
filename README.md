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

bash
Copy code
git clone https://github.com/yourusername/ML_CustomerChurn.git
Navigate to the project directory:

bash
Copy code
cd ML_CustomerChurn
Install the necessary dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Training the Model: Train the model using historical customer data by following the steps in training.py.

Making Predictions: You can use the Streamlit application to upload a CSV file containing customer data and predict churn.

bash
Copy code
streamlit run customer_churn.py
Model Deployment: The trained model is saved as xgb_model.pkl and can be used for batch predictions or real-time predictions.

Evaluation
Model Performance: After training, evaluate the model using various metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

python
Copy code
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
Classification Report
Precision	Recall	F1-Score	Support
0	0.96	0.98	0.97	849
1	0.86	0.78	0.82	163
Accuracy	-	-	-	1012
Macro avg	0.91	0.88	0.89	1012
Weighted avg	0.94	0.94	0.94	1012
Model Accuracy: 0.94
