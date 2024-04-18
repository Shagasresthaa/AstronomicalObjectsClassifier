#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[3]:


galaxy_data = np.genfromtxt(r'C:\Users\Nihal\OneDrive\Documents\photometric-data_03-29-2024_photoMetricData_combined_photometric_data_GALAXY.csv', delimiter=',', skip_header=1)


# In[4]:


qso_data = np.genfromtxt(r'C:\Users\Nihal\OneDrive\Documents\photometric-data_03-29-2024_photoMetricData_combined_photometric_data_QSO.csv', delimiter=',', skip_header=1)


# In[5]:


# Sample 1000 points from each class
galaxy_sample = galaxy_data[:1000]
qso_sample = qso_data[:1000]


# In[6]:


# Extract u-g and r-z magnitudes for galaxies and QSOs
u_g_galaxy = galaxy_sample[:, 0]  # Assuming u-g magnitudes are in the first column
r_z_galaxy = galaxy_sample[:, 1]  # Assuming r-z magnitudes are in the second column

u_g_qso = qso_sample[:, 0]  # Assuming u-g magnitudes are in the first column
r_z_qso = qso_sample[:, 1]  # Assuming r-z magnitudes are in the second column


# In[10]:


# Ploting the color-color diagram for galaxies and QSOs
plt.figure(figsize=(8, 6))
plt.scatter(u_g_galaxy, r_z_galaxy, s=10, c='blue', alpha=0.5, label='Galaxies')
plt.scatter(u_g_qso, r_z_qso, s=10, c='red', alpha=0.5, label='QSOs')

plt.xlabel('u-g Magnitude')
plt.ylabel('r-z Magnitude')
plt.title('u-g vs r-z Color-Color Diagram (Sample)')
plt.grid(True)
plt.legend()
plt.show()


# In[ ]:




