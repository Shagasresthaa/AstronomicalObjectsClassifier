#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import itertools

# Define function for plotting confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, cmap=plt.cm.Blues):
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Normalized confusion matrix'
    else:
        title = 'Confusion matrix'

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Load photometric dataset for galaxies
galaxy_data = np.genfromtxt(r'C:\Users\Nihal\OneDrive\Documents\photometric-data_03-29-2024_photoMetricData_combined_photometric_data_GALAXY.csv', delimiter=',', skip_header=1)

# Load photometric dataset for QSOs
qso_data = np.genfromtxt(r'C:\Users\Nihal\OneDrive\Documents\photometric-data_03-29-2024_photoMetricData_combined_photometric_data_QSO.csv', delimiter=',', skip_header=1)

# Create labels for galaxies (class 0) and QSOs (class 1)
galaxy_labels = np.zeros(len(galaxy_data))
qso_labels = np.ones(len(qso_data))

# Combine datasets and labels
X = np.concatenate((galaxy_data, qso_data), axis=0)
y = np.concatenate((galaxy_labels, qso_labels))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict labels for the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
plot_confusion_matrix(y_test, y_pred, classes=['Galaxies', 'QSOs'], normalize=True)
plt.show()


# In[5]:


print(confusion_matrix(y_test, y_pred))


# In[ ]:




