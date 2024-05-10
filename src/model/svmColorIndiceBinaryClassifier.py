from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import logging
from datetime import datetime
import os
from joblib import dump
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Logging setup with timestamped filenames
log_directory = "logs/model_logs"
os.makedirs(log_directory, exist_ok=True)  # Ensure the log directory exists
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"svm_color_indices_training_log_{timestamp}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def trainSVMClassifier(df):
    logging.info("Training with RBF kernel initiated.....")

    X = df.drop('class', axis=1)
    y = df['class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

    param_grid = {
        'C': np.logspace(-3, 3, 7),
        'gamma': np.logspace(-3, 2, 6)
    }

    svm_model = SVC(kernel='rbf')
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logging.info(f"Best Parameters for RBF kernel: {grid_search.best_params_}")
    logging.info(f"Best Cross-Validation Score for RBF kernel: {grid_search.best_score_:.2f}")

    # Logging detailed grid search results
    logging.info("Grid Search Detailed Results:")
    results = grid_search.cv_results_
    for mean_score, std_score, params in zip(results['mean_test_score'], results['std_test_score'], results['params']):
        logging.info(f"Mean CV Score: {mean_score:.3f} (+/-{std_score * 2:.3f}) for {params}")

    y_train_pred = grid_search.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    logging.info(f"Training Set Accuracy with RBF kernel: {train_accuracy:.2f}")

    y_pred = grid_search.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"Test Set Accuracy with RBF kernel: {test_accuracy:.2f}")

    # Log the confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    logging.info("Confusion Matrix:\n" + str(conf_matrix))

    classification_rep = classification_report(y_test, y_pred, output_dict=True)
    logging.info("Classification Report with RBF kernel:\n" + str(classification_rep))

    model_filename = f"models/svm_rbf_best_{timestamp}.joblib"
    dump(grid_search.best_estimator_, model_filename)
    logging.info(f"Best model with RBF kernel saved to {model_filename}")

    scaler_filename = f"models/scaler_{timestamp}.joblib"
    dump(scaler, scaler_filename)
    logging.info(f"Scaler saved to {scaler_filename}")

    metrics = {
        'best_parameters': grid_search.best_params_,
        'best_cv_score': grid_search.best_score_,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'confusion_matrix': conf_matrix.tolist(), 
        'classification_report': classification_rep
    }

    return grid_search.best_estimator_, scaler, metrics

def automateTraining(classes):
    logging.info("Training session initiated.....")
    procPhotoDataPath = "data/processed/photoMetricData/"
    combinationDataframe = []

    for astroClass in classes:
        procFile = os.path.join(procPhotoDataPath, f"combined_photometric_data_{astroClass}.csv")
        df = pd.read_csv(procFile, comment='#')
        combinationDataframe.append(df.iloc[:25000]) 

    combined_df = pd.concat(combinationDataframe, ignore_index=True)
    combined_df['u-g'] = combined_df['u'] - combined_df['g']
    combined_df['g-r'] = combined_df['g'] - combined_df['r']
    combined_df = combined_df[['class', 'u-g', 'g-r']].copy()

    logging.info(f"Combined DataFrame stats:\n{combined_df.describe()}")
    logging.info(f"Combined DataFrame stats:\n{combined_df.head()}")

    model, scaler, metrics = trainSVMClassifier(combined_df)
    logging.info(f"Model and scaler training completed. Detailed metrics:\n{metrics}")

if __name__ == "__main__":
    classList = ["GALAXY", "QSO"]
    automateTraining(classList)
