#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load photometric dataset for galaxies
galaxy_data = np.genfromtxt(r'C:\Users\Nihal\OneDrive\Documents\photometric-data_03-29-2024_photoMetricData_combined_photometric_data_GALAXY.csv', delimiter=',', skip_header=1)
# Extract color indices for galaxies
galaxy_color_indices = galaxy_data[:, [1, 2]]  # Assuming columns 1 and 2 correspond to u-g and r-z

# Load photometric dataset for QSOs
qso_data = np.genfromtxt(r'C:\Users\Nihal\OneDrive\Documents\photometric-data_03-29-2024_photoMetricData_combined_photometric_data_QSO.csv', delimiter=',', skip_header=1)
# Extract color indices for QSOs
qso_color_indices = qso_data[:, [1, 2]]  # Assuming columns 1 and 2 correspond to u-g and r-z

# Combine color indices and labels
X = np.concatenate((galaxy_color_indices, qso_color_indices), axis=0)
y = np.concatenate((np.zeros(len(galaxy_color_indices)), np.ones(len(qso_color_indices))))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning with Grid Search
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
clf = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train a decision tree classifier with the best parameters
best_clf = DecisionTreeClassifier(random_state=42, **best_params)
best_clf.fit(X_train, y_train)

# Predict labels for the test set using the best classifier
y_pred = best_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy after Hyperparameter Tuning:", accuracy)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
conf_mat = confusion_matrix(y_test, y_pred)
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ['Galaxies', 'QSOs'])
plt.yticks([0, 1], ['Galaxies', 'QSOs'])
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Display values on confusion matrix
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        plt.text(j, i, format(conf_mat[i, j], 'd'), horizontalalignment="center", color="white" if conf_mat[i, j] > conf_mat.max() / 2 else "black")

plt.show()


# In[1]:


print(confusion_matrix(y_test, y_pred))


# In[ ]:




