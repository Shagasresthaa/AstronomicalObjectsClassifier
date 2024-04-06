import pandas as pd
import logging
from datetime import datetime
import subprocess
import os

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"photometric_data_prep_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def photoMetricDataExtract(astroClass):
    photoMetricDataPathDir = {"GALAXY": "data/raw/photometricDataExtract/galaxy/", "STAR": "data/raw/photometricDataExtract/star/", "QSO": "data/raw/photometricDataExtract/qso/"}
    photoMetricClassPath = photoMetricDataPathDir.get(astroClass)

    # Command will work for linux and maybe a mac but not sure if it will work on windows
    # Replace command in windows with appropriate one if it doesnt work
    cmd = f"ls {photoMetricClassPath}/photometric_data_batch_{astroClass}_*.csv | wc -l"

    # Running the command and capturing the output
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    fileCount = int(result.stdout.strip())
    logging.info(f"Total Files for class {astroClass}: {fileCount}")

    dataframes = []

    for i in range(1, fileCount + 1):
        photoMetricFilePath = photoMetricClassPath + f'photometric_data_batch_{astroClass}_{i}.csv'
        logging.info(f"Reading and Appending file '{photoMetricFilePath}' for class '{astroClass}'....")
        df = pd.read_csv(photoMetricFilePath, comment='#')
        dataframes.append(df)
        logging.info(f"Reading and Appending file '{photoMetricFilePath}' for class '{astroClass}' done....")

    try:
        # Concatenate all DataFrames in the list
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Save the combined DataFrame to a new CSV file
        procPhotoMetricDataPath = "data/processed/photoMetricData/"
        procPhotoMetricFilePath = f"{procPhotoMetricDataPath}/combined_photometric_data_{astroClass}.csv"
        logging.info(f"Writing Combination File {procPhotoMetricDataPath}...")
        combined_df.to_csv(procPhotoMetricFilePath, index=False)
        logging.info(f"Written Combination File {procPhotoMetricDataPath}...")
    except Exception as e:
            logging.exception(e)

def automateFetching(classes):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Processing PhotoMetric data initiated.....{timestamp}")

    for astroClass in classes:
        try:
            logging.info(f"Processing PhotoMetric data for: {astroClass}")
            photoMetricDataExtract(astroClass)
            logging.info(f"Processing PhotoMetric data done for: {astroClass}")
        except Exception as e:
            logging.exception(f"Failed to fetch Photometric data.....")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Processing PhotoMetric data completed.....{timestamp}")

if __name__ == "__main__":
    classList = ["GALAXY", "STAR", "QSO"]
    automateFetching(classList)