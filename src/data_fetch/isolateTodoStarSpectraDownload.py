import requests as reqObj
import pandas as pd
import logging
from datetime import datetime
import os
import time
import threading

from configs.sdssApiEndpoints import SDSS_OBJ_SQL_SEARCH_BASE, SAS_SPEC_FITS_FETCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_fetch_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_class_data_aggregate_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

starClasses = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

def checkFileExistance(csvFilePath):
    if os.path.exists(csvFilePath):
        logging.info(f"File already exists in raw spectra so skipping at path {csvFilePath}")
        return True
    else:
        logging.info(f"File doesnt exists in raw spectra at path {csvFilePath} so adding to TODO downloads")
        return False

def verifyAndCreateNewMasterFile():
    logging.info("Starting to fetch Star Class Data")
    for starClass in starClasses:
        try:
            logging.info(f"Verifying and Aggregating data for: {starClass}")
            starDataPath = f'data/raw/new_star_class_data/{starClass}_star_class_1.csv'
            fitsSpectraPath = 'data/raw/fits_files/spectral_fits/star/'
            cols = ['objid','plate','mjd','fiberid','effectiveTemperature']
            colsExist = ['objid','plate','mjd','fiberid','effectiveTemperature', 'fitsFilePath']
            starClassData = pd.read_csv(starDataPath, comment='#')
            fileExistsRows = []
            fileNotExistsRows = []
            logging.info("Checking FITS data availability for class")
            for index, row in starClassData.iterrows():
                if checkFileExistance(f"{fitsSpectraPath}{row['objid']}_spec.fits"):
                    row['fitsFilePath'] = f"{fitsSpectraPath}{row['objid']}_spec.fits"
                    fileExistsRows.append(row)
                else:
                    fileNotExistsRows.append(row)
            
            existingData = pd.DataFrame(fileExistsRows, columns=colsExist)
            needToDownloadData = pd.DataFrame(fileNotExistsRows, columns=cols)

            existingData.to_csv(f"data/raw/new_star_class_data/{starClass}_existing_master_data.csv")
            needToDownloadData.to_csv(f"data/raw/new_star_class_data/{starClass}_non_existing_master_data.csv")
            logging.info(f"Done Verifying and Aggregating data for: {starClass}")

        except Exception as e:
            logging.exception(f"Failed to fetch data for {starClass}")

    logging.info("Completed verifying and fetching Star Class Data")
    
if __name__ == "__main__":
    verifyAndCreateNewMasterFile()