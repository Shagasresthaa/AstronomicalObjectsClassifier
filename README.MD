# Astronomical Objects Classification with Convolutional Neural Networks and SDSS data

## Introduction

This project aims to classify astronomical objects using data from the Sloan Digital Sky Survey (SDSS) and Convolutional Neural Networks (CNN). It leverages python and popular machine learning libraries to accurately classify galaxies, stars, and quasars based on their image and spectral data.

## Features

- Utilizes SDSS DR18 data.
- Implements a CNN model for classification.
- Provides a pipeline for data preprocessing, training, and evaluation.

## Project Prerequisites

Before you begin, ensure you have met the following requirements:

Python: This project uses Python version 3.11.8. Download Python
TensorFlow: TensorFlow 2.x is required for model building and training. Ensure compatibility with Python 3.11.8.

## Installation

Clone the Project:

- git clone git@github.com:Shagasresthaa/AstronomicalObjectsClassifier.git
- cd AstronomicalObjectsClassifier

(As per new Github security policies, you will need to either setup an SSH key and setup in your account or setup an access key [SSH setup is rather simple and wouldnt have any need to regenerate periodically like access key] in order to do any code movement.

Heres a quick link to the tutorial to do so: https://www.youtube.com/watch?v=X40b9x9BFGo

Set Up a Virtual Environment (Optional but recommended):

(localEnv is the name i used, feel free to use any name but please make sure to add it in .gitignore as you dont want them in repo)

- python -m venv localEnv
- source localEnv/bin/activate # On Windows use `localEnv\Scripts\activate`

## Install Required Libraries:

Navigate to the root of the project and install the required packages using pip with following command:

pip install -r requirements.txt

# Dataset Fetch Scripts Guide

## CSV Data fetch

Navigate to the root of the project and run the following command which will fetch the latest data from SDSS API

python3 -m src.data_fetch.fetchCsvSearchTargetData

## Additional Tools

- Jupyter Notebook: For exploratory data analysis and model prototyping.
- Git: For version control.
- SDSS Data Access: Familiarize yourself with any API requirements if you intend to download data programmatically.
