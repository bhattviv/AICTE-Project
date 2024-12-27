# -*- coding: utf-8 -*-
"""diabetes_disease(week_3).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RNnW4evNwa2dhw5d4r8SCxXeFCMUhWUk

# Predicting diabetes disease using Machine Learning
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
# Load the Iris dataset
df=pd.read_csv('E:\\datasets\\diabetes.csv')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

import matplotlib.pyplot as plt
import seaborn as sns

"""### Importing the dataset

### Shape of the dataset (Rows, Columns)
"""

df.shape

"""### Head of the dataset"""

df.head()

"""<br><br>
# Exploratory Data analysis
<br>

### Renaming columns
"""

df.rename(columns ={'age':'Age','sex':'Sex','cp':'Chest_pain','trestbps':'Resting_blood_pressure','chol':'Cholesterol','fbs':'Fasting_blood_sugar',
                    'restecg':'ECG_results','thalach':'Maximum_heart_rate','exang':'Exercise_induced_angina','oldpeak':'ST_depression','slope':'ST_slope','ca':'Major_vessels',
                   'thal':'Thalassemia_types','target':'Heart_disease'}, inplace = True)

# View of the Renamed Dataframe
df.head()

df.tail()

"""### Information about the data"""

list(df.columns.values)

df.info()

"""### Description about the dataset"""

df.describe()

"""### Are there any missing values?"""

df.isna().sum()

df.groupby('Outcome').mean()

df['Outcome'].value_counts()

"""### Correlation matrix & Matrix Visualisation"""

df.corr()

# Let's make our correlation matrix visual
corr_matrix=df.corr()
fig,ax=plt.subplots(figsize=(15,10))
ax=sns.heatmap(corr_matrix,
               annot=True,
               linewidths=0.5,
               fmt=".2f"
              )

X=df.drop(['Outcome'],axis=1)
y=df['Outcome']

print(X)

print(y)

import matplotlib.pyplot as plt

# Plot histograms for all numerical features
df.hist(bins=20, figsize=(15, 10), edgecolor='black')
plt.suptitle("Histograms for Each Feature", fontsize=16)
plt.show()

import seaborn as sns

# Pairplot to visualize pairwise relationships between features
sns.pairplot(df, diag_kind='kde')
plt.suptitle("Pairplot for Feature Relationships", y=1.02, fontsize=16)
plt.show()

# Plot boxplots for each numerical feature
plt.figure(figsize=(15, 10))
df.boxplot()
plt.title("Boxplots for Each Feature", fontsize=16)
plt.xticks(rotation=45)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, stratify=y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

from sklearn.svm import SVC # You've already done this previously

classifier = SVC(kernel='linear')

classifier.fit(X_train, y_train)

# accuracy score on the training data
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, y_train)

print('Accuracy score of the training data : ', training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, y_test)

print('Accuracy score of the test data : ', test_data_accuracy)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Split into training and test sets
# Standardizing the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model
logistic_model = LogisticRegression(random_state=42, max_iter=1000)
logistic_model.fit(X_train_scaled, y_train)

# Predicting on test data
y_pred = logistic_model.predict(X_test_scaled)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Initialize the Decision Tree model
dt_model = DecisionTreeClassifier(random_state=42)

# Train the model
dt_model.fit(X_train_scaled, y_train)

# Predict on test data
y_pred_dt = dt_model.predict(X_test_scaled)

# Evaluate the model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

# Optional: Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# Random Forest with Grid Search
rf = RandomForestClassifier(random_state=42)
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}
grid_rf = GridSearchCV(rf, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

# Best estimator and evaluation
best_rf = grid_rf.best_estimator_
rf_predictions = best_rf.predict(X_test)
print("Random Forest Classification Report:\n", classification_report(y_test, rf_predictions))

from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Support Vector Machine
svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

# Evaluate SVM
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions))

import xgboost as xgb
from sklearn.metrics import classification_report

# XGBoost Model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)

# Evaluate XGBoost
print("XGBoost Classification Report:\n", classification_report(y_test, xgb_predictions))

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Multi-Layer Perceptron (Neural Network)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)

# Evaluate MLP
print("MLP Classification Report:\n", classification_report(y_test, mlp_predictions))

from sklearn.metrics import roc_auc_score

# Compare models using ROC-AUC
models = {'Random Forest': best_rf, 'SVM': svm, 'XGBoost': xgb_model, 'MLP': mlp}
for name, model in models.items():
    y_prob = model.predict_proba(X_test)[:, 1]  # Predict probabilities
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name} ROC-AUC Score: {auc}")

"""**Hyperparameter Tuning and Evaluation**"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
import shap
import numpy as np

# Logistic Regression Model with Hyperparameter Tuning
param_grid_lr = {'C': [0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['liblinear', 'lbfgs']}
grid_lr = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid_lr, cv=5, scoring='accuracy')
grid_lr.fit(X_train_scaled, y_train)

# Best Logistic Regression Model
tuned_logistic_model = grid_lr.best_estimator_
y_pred = tuned_logistic_model.predict(X_test_scaled)

# Evaluating the tuned model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy after Hyperparameter Tuning: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# SHAP Interpretation
explainer = shap.Explainer(tuned_logistic_model, X_train_scaled)
shap_values = explainer(X_test_scaled)

# Visualizing SHAP values
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

# Improved Accuracy and Interpretation completed

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import shap

# Decision Tree Hyperparameter Tuning
param_grid_dt = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, scoring='accuracy')
grid_dt.fit(X_train_scaled, y_train)

# Best Decision Tree Model
tuned_dt_model = grid_dt.best_estimator_
y_pred_dt = tuned_dt_model.predict(X_test_scaled)

# Evaluate the tuned model
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Accuracy after Hyperparameter Tuning: {accuracy_dt * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))

# SHAP Interpretation
explainer = shap.Explainer(tuned_dt_model, X_train_scaled)
shap_values = explainer(X_test)

# Visualizing SHAP values
shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = classifier.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

import pickle
filename = 'diabetes_model.sav'
pickle.dump(best_rf, open(filename, 'wb'))

loaded_model = pickle.load(open('diabetes_model.sav', 'rb'))

input_data = (5,166,72,19,175,25.8,0.587,51)

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The person is diabetic')

for column in X.columns:
  print(column)