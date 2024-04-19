# Astronomical Objects Classification with Convolutional Neural Networks and SDSS data

## Introduction

This project aims to classify astronomical objects using data from the Sloan Digital Sky Survey (SDSS) and Convolutional Neural Networks (CNN). It leverages python and popular machine learning libraries to accurately classify galaxies, stars, and quasars based on their image and spectral data.

## Features

- Utilizes SDSS DR18 data.
- Implements a multi-model approach to classify astronomical objects
- Uses a binary classification CNN for classifying if an object is a star or not
- Uses a Random Forest model to classify if an object is a Galaxy or a QSO
- Uses a 1-D Time Series Analysis CNN Classifier to classify any given star into its major class type based on the Morgan–Keenan System
- Provides a pipeline for data preprocessing, training, and evaluation

## Project Prerequisites

Before you begin, ensure you have met the following requirements:

Python: This project uses Python version 3.11.8. Download Python
TensorFlow: TensorFlow 2.x is required for model building and training. Ensure compatibility with Python 3.11.8.

## Installation

Clone the Project:

- git clone git@github.com:Shagasresthaa/AstronomicalObjectsClassifier.git
- cd AstronomicalObjectsClassifier

(As per new Github security policies, you will need to either setup an SSH key and do Github SSH setup or setup a rotating temporary access key [SSH setup is rather simple and wouldn't have any need to regenerate periodically like access key] to do any code movement.

Here's a quick link to the tutorial to do so: https://www.youtube.com/watch?v=X40b9x9BFGo

Set Up a Virtual Environment (Optional but recommended):

(localEnv is the name used for this repo but make sure to name the local environment appropriately if you are using and add to .gitignore)

- python -m venv localEnv
- source localEnv/bin/activate # On Windows use `localEnv\Scripts\activate`

## Install Required Libraries:

Navigate to the root of the project and install the required packages using pip with the following command:

pip install -r requirements.txt

# Scripts Guide

Navigate to the root of the project and run the commands as per specifications and your requirements

python3 -m path.to.directory

E.g.: To run star classifier Model at path src/model/starClassClassifier.py
    "python3 -m src.model.starClassClassifier" or "python -m src.model.starClassClassifier"

## Script Directory paths might need to be altered based on the operating system in use (currently all paths used work well in a Linux environment)

## Datasets used for the project (Fair Warning the datasets are massive in size): http://astronomical-objects-classifier-ml-cs-5821-sp-2024-team-nasa.s3-website-us-east-1.amazonaws.com/#


## Additional Tools

- Jupyter Notebook: For exploratory data analysis and model prototyping.
- Git: For version control.
- SDSS Data Access: Familiarize yourself with any API requirements if you intend to download data programmatically.
