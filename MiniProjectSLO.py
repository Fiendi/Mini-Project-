#!/usr/bin/env python
# coding: utf-8

# In[12]:


# Nama : Tia Fiendi A
# NIM : 1101204055


# In[10]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load data
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPA1.xlsx")

# Split features and target variable
X = data.drop('kelas', axis=1)
y = data['kelas']

# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)

    return sensitivity, specificity, accuracy, f1_score

# SVM Classifier
def svm_classifier(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    sensitivity, specificity, accuracy, f1_score = calculate_metrics(y_test, y_pred)

    print("SVM Performance Metrics:")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)
    print("F1-Score:", f1_score)

    return svm

# K-NN Classifier
def knn_classifier(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    sensitivity, specificity, accuracy, f1_score = calculate_metrics(y_test, y_pred)

    print("\nK-NN Performance Metrics:")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)
    print("F1-Score:", f1_score)

    return knn

# 10-fold Cross Validation
def cross_validation(clf, X, y):
    cv_scores = cross_val_score(clf, X, y, cv=10)
    print("\nCross Validation Accuracy (10-fold CV):", np.mean(cv_scores))

# Split data for 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# SVM Classifier - 75% training, 25% testing
svm_model = svm_classifier(X_train, X_test, y_train, y_test)

# K-NN Classifier - 75% training, 25% testing
knn_model = knn_classifier(X_train, X_test, y_train, y_test)

# Cross Validation
print("\nPerforming 10-fold Cross Validation:")
cross_validation(svm_model, X, y)
cross_validation(knn_model, X, y)

# Split data for 50% training, 25% validation, and 25% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SVM Classifier - 50% training, 25% validation, and 25% testing
svm_model = svm_classifier(X_train, X_test, y_train, y_test)

# K-NN Classifier - 50% training, 25% validation, and 25% testing
knn_model = knn_classifier(X_train, X_test, y_train, y_test)



# In[13]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load data
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPA2.xlsx")

# Split features and target variable
X = data.drop('kelas', axis=1)
y = data['kelas']

# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)

    return sensitivity, specificity, accuracy, f1_score

# SVM Classifier
def svm_classifier(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    sensitivity, specificity, accuracy, f1_score = calculate_metrics(y_test, y_pred)

    print("SVM Performance Metrics:")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)
    print("F1-Score:", f1_score)

    return svm

# K-NN Classifier
def knn_classifier(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    sensitivity, specificity, accuracy, f1_score = calculate_metrics(y_test, y_pred)

    print("\nK-NN Performance Metrics:")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)
    print("F1-Score:", f1_score)

    return knn

# 10-fold Cross Validation
def cross_validation(clf, X, y):
    cv_scores = cross_val_score(clf, X, y, cv=10)
    print("\nCross Validation Accuracy (10-fold CV):", np.mean(cv_scores))

# Split data for 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# SVM Classifier - 75% training, 25% testing
svm_model = svm_classifier(X_train, X_test, y_train, y_test)

# K-NN Classifier - 75% training, 25% testing
knn_model = knn_classifier(X_train, X_test, y_train, y_test)

# Cross Validation
print("\nPerforming 10-fold Cross Validation:")
cross_validation(svm_model, X, y)
cross_validation(knn_model, X, y)

# Split data for 50% training, 25% validation, and 25% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SVM Classifier - 50% training, 25% validation, and 25% testing
svm_model = svm_classifier(X_train, X_test, y_train, y_test)

# K-NN Classifier - 50% training, 25% validation, and 25% testing
knn_model = knn_classifier(X_train, X_test, y_train, y_test)



# In[14]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

# Load data
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPA3.xlsx")

# Split features and target variable
X = data.drop('kelas', axis=1)
y = data['kelas']

# Function to calculate performance metrics
def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = accuracy_score(y_true, y_pred)
    f1_score = 2 * (sensitivity * specificity) / (sensitivity + specificity)

    return sensitivity, specificity, accuracy, f1_score

# SVM Classifier
def svm_classifier(X_train, X_test, y_train, y_test):
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)

    sensitivity, specificity, accuracy, f1_score = calculate_metrics(y_test, y_pred)

    print("SVM Performance Metrics:")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)
    print("F1-Score:", f1_score)

    return svm

# K-NN Classifier
def knn_classifier(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    sensitivity, specificity, accuracy, f1_score = calculate_metrics(y_test, y_pred)

    print("\nK-NN Performance Metrics:")
    print("Sensitivity:", sensitivity)
    print("Specificity:", specificity)
    print("Accuracy:", accuracy)
    print("F1-Score:", f1_score)

    return knn

# 10-fold Cross Validation
def cross_validation(clf, X, y):
    cv_scores = cross_val_score(clf, X, y, cv=10)
    print("\nCross Validation Accuracy (10-fold CV):", np.mean(cv_scores))

# Split data for 75% training and 25% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# SVM Classifier - 75% training, 25% testing
svm_model = svm_classifier(X_train, X_test, y_train, y_test)

# K-NN Classifier - 75% training, 25% testing
knn_model = knn_classifier(X_train, X_test, y_train, y_test)

# Cross Validation
print("\nPerforming 10-fold Cross Validation:")
cross_validation(svm_model, X, y)
cross_validation(knn_model, X, y)

# Split data for 50% training, 25% validation, and 25% testing
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.5, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# SVM Classifier - 50% training, 25% validation, and 25% testing
svm_model = svm_classifier(X_train, X_test, y_train, y_test)

# K-NN Classifier - 50% training, 25% validation, and 25% testing
knn_model = knn_classifier(X_train, X_test, y_train, y_test)



# In[23]:


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data from Excel
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPC1.xlsx")

# Split features and target variable
X = data.drop(columns=['kelas'])
y = data['kelas']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predictions on testing set
svm_predictions = svm_classifier.predict(X_test)

# Evaluate performance
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Print SVM performance metrics
print("Support Vector Machine (SVM) Performance:")
print("Accuracy:", svm_accuracy)
print("Specificity:", svm_precision)
print("Sensitivity:", svm_recall)
print("F1-Score:", svm_f1)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predictions on testing set
knn_predictions = knn_classifier.predict(X_test)

# Evaluate performance
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

# Print KNN performance metrics
print("\nK-Nearest Neighbors (KNN) Performance:")
print("Accuracy:", knn_accuracy)
print("Specificity:", knn_precision)
print("Sensitivity:", knn_recall)
print("F1-Score:", knn_f1)

# 10-fold cross-validation
svm_cv_scores = cross_val_score(svm_classifier, X_scaled, y, cv=10)
knn_cv_scores = cross_val_score(knn_classifier, X_scaled, y, cv=10)

# Print cross-validation scores
print("\n10-Fold Cross-Validation Scores:")
print("SVM Mean Accuracy:", np.mean(svm_cv_scores))
print("KNN Mean Accuracy:", np.mean(knn_cv_scores))


# In[22]:


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data from Excel
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPC2.xlsx")

# Split features and target variable
X = data.drop(columns=['kelas'])
y = data['kelas']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predictions on testing set
svm_predictions = svm_classifier.predict(X_test)

# Evaluate performance
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Print SVM performance metrics
print("Support Vector Machine (SVM) Performance:")
print("Accuracy:", svm_accuracy)
print("Specificity:", svm_precision)
print("Sensitivity:", svm_recall)
print("F1-Score:", svm_f1)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predictions on testing set
knn_predictions = knn_classifier.predict(X_test)

# Evaluate performance
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

# Print KNN performance metrics
print("\nK-Nearest Neighbors (KNN) Performance:")
print("Accuracy:", knn_accuracy)
print("Specificity:", knn_precision)
print("Sensitivity:", knn_recall)
print("F1-Score:", knn_f1)

# 10-fold cross-validation
svm_cv_scores = cross_val_score(svm_classifier, X_scaled, y, cv=10)
knn_cv_scores = cross_val_score(knn_classifier, X_scaled, y, cv=10)

# Print cross-validation scores
print("\n10-Fold Cross-Validation Scores:")
print("SVM Mean Accuracy:", np.mean(svm_cv_scores))
print("KNN Mean Accuracy:", np.mean(knn_cv_scores))


# In[27]:


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data from Excel
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPC3.xlsx")

# Split features and target variable
X = data.drop(columns=['kelas'])
y = data['kelas']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predictions on testing set
svm_predictions = svm_classifier.predict(X_test)

# Evaluate performance
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Print SVM performance metrics
print("Support Vector Machine (SVM) Performance:")
print("Accuracy:", svm_accuracy)
print("Specificity:", svm_precision)
print("Sensitivity:", svm_recall)
print("F1-Score:", svm_f1)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predictions on testing set
knn_predictions = knn_classifier.predict(X_test)

# Evaluate performance
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

# Print KNN performance metrics
print("\nK-Nearest Neighbors (KNN) Performance:")
print("Accuracy:", knn_accuracy)
print("Specificity:", knn_precision)
print("Sensitivity:", knn_recall)
print("F1-Score:", knn_f1)

# 10-fold cross-validation
svm_cv_scores = cross_val_score(svm_classifier, X_scaled, y, cv=10)
knn_cv_scores = cross_val_score(knn_classifier, X_scaled, y, cv=10)

# Print cross-validation scores
print("\n10-Fold Cross-Validation Scores:")
print("SVM Mean Accuracy:", np.mean(svm_cv_scores))
print("KNN Mean Accuracy:", np.mean(knn_cv_scores))


# In[26]:


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data from Excel
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPC4.xlsx")

# Split features and target variable
X = data.drop(columns=['kelas'])
y = data['kelas']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predictions on testing set
svm_predictions = svm_classifier.predict(X_test)

# Evaluate performance
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Print SVM performance metrics
print("Support Vector Machine (SVM) Performance:")
print("Accuracy:", svm_accuracy)
print("Specificity:", svm_precision)
print("Sensitivity:", svm_recall)
print("F1-Score:", svm_f1)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predictions on testing set
knn_predictions = knn_classifier.predict(X_test)

# Evaluate performance
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

# Print KNN performance metrics
print("\nK-Nearest Neighbors (KNN) Performance:")
print("Accuracy:", knn_accuracy)
print("Specificity:", knn_precision)
print("Sensitivity:", knn_recall)
print("F1-Score:", knn_f1)

# 10-fold cross-validation
svm_cv_scores = cross_val_score(svm_classifier, X_scaled, y, cv=10)
knn_cv_scores = cross_val_score(knn_classifier, X_scaled, y, cv=10)

# Print cross-validation scores
print("\n10-Fold Cross-Validation Scores:")
print("SVM Mean Accuracy:", np.mean(svm_cv_scores))
print("KNN Mean Accuracy:", np.mean(knn_cv_scores))


# In[25]:


# Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data from Excel
data = pd.read_excel("E:\\KULIAH\\SEMESTER 2 S2\\SLO\\CPC5.xlsx")

# Split features and target variable
X = data.drop(columns=['kelas'])
y = data['kelas']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Support Vector Machine (SVM) Classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Predictions on testing set
svm_predictions = svm_classifier.predict(X_test)

# Evaluate performance
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_precision = precision_score(y_test, svm_predictions, average='weighted')
svm_recall = recall_score(y_test, svm_predictions, average='weighted')
svm_f1 = f1_score(y_test, svm_predictions, average='weighted')

# Print SVM performance metrics
print("Support Vector Machine (SVM) Performance:")
print("Accuracy:", svm_accuracy)
print("Specificity:", svm_precision)
print("Sensitivity :", svm_recall)
print("F1-Score:", svm_f1)

# K-Nearest Neighbors (KNN) Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)

# Predictions on testing set
knn_predictions = knn_classifier.predict(X_test)

# Evaluate performance
knn_accuracy = accuracy_score(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions, average='weighted')
knn_recall = recall_score(y_test, knn_predictions, average='weighted')
knn_f1 = f1_score(y_test, knn_predictions, average='weighted')

# Print KNN performance metrics
print("\nK-Nearest Neighbors (KNN) Performance:")
print("Accuracy:", knn_accuracy)
print("Specificity:", knn_precision)
print("Sensitivity:", knn_recall)
print("F1-Score:", knn_f1)

# 10-fold cross-validation
svm_cv_scores = cross_val_score(svm_classifier, X_scaled, y, cv=10)
knn_cv_scores = cross_val_score(knn_classifier, X_scaled, y, cv=10)

# Print cross-validation scores
print("\n10-Fold Cross-Validation Scores:")
print("SVM Mean Accuracy:", np.mean(svm_cv_scores))
print("KNN Mean Accuracy:", np.mean(knn_cv_scores))


# In[ ]:





# In[ ]:




