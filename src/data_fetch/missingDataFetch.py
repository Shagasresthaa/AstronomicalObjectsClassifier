import requests as reqObj
import pandas as pd
import logging
from datetime import datetime
import os
import time
import threading

from configs.sdssApiEndpoints import SAS_SPEC_FITS_FETCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_fetch_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"missing_star_data_download_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

#starClasses = {'A': 66, 'G': 867, 'O': 2790, 'B': 2000}
#starClasses = {'B': 1000, 'A': 1000, 'F': 1000, 'G': 1000}
# Init to len required for dataset
starClasses = {'O': 2897, 'B': 5000, 'A': 5000, 'F': 5000, 'G': 5000, 'K': 4000, 'M': 4000}
starRawLoc = 'data/raw/fits_files/spectral_fits/star/'

def fitsDownloader(url, filePath, max_attempts=100):
    attempt = 1
    wait_time = 0.5  # initial delay in seconds

    while attempt <= max_attempts:
        try:
            with reqObj.get(url, stream=True) as response:
                response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
                with open(filePath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:  # filter out keep-alive chunks
                            f.write(chunk)
            logging.info(f"Successfully written {filePath} on attempt {attempt}.")
            return
        except reqObj.exceptions.RequestException as e:
            logging.error(f"Attempt {attempt} failed with error: {e}")
            time.sleep(wait_time)  # Wait before retrying
            attempt += 1
            if attempt > max_attempts:
                logging.error(f"All {max_attempts} attempts failed. Could not download the file.")
                raise

            # Increase wait_time for the next attempt, capped at 300 seconds (5 minutes)
            wait_time = min(wait_time * 2, 300)  # Double the wait time but cap at 300 seconds

def checkFileExistance(csvFilePath):
    exists = os.path.exists(csvFilePath)
    file_size = os.path.getsize(csvFilePath) if exists else 0
    if exists and file_size > 0:
        logging.info(f"File already exists and is not empty at path {csvFilePath}")
        return True
    else:
        logging.info(f"File does not exist or is empty at path {csvFilePath}")
        return False
    
def downloadMissingData(objid, plate, mjd, fiberid):
    logging.info(f"Starting object download: {objid}")
    finalRawFilePath = f"{starRawLoc}{objid}_spec.fits"
    if checkFileExistance(finalRawFilePath):
        sasFitsFetchUrl = f"{SAS_SPEC_FITS_FETCH_BASE}plateid={plate}&mjd={mjd}&fiberid={fiberid}"
        fitsDownloader(sasFitsFetchUrl, finalRawFilePath)
        logging.info(f"Completed object download: {objid}")
        logging.info("Initiating Sleep for 0.5 secs to preserve api rate limits")
    else:
        logging.info("Object already downloaded and ready")
    return finalRawFilePath

if __name__ == "__main__":
    logging.info("Initiating Missing Data Download")
    for starClass, size in starClasses.items():
        existsDf = pd.read_csv(f"data/raw/new_star_class_data/{starClass}_existing_master_data.csv")
        nonExistsDf = pd.read_csv(f"data/raw/new_star_class_data/{starClass}_non_existing_master_data.csv")
        existsLen = len(existsDf)

        existDfNewRows = []
        if(existsLen >= size):
            logging.info(f"Data is already available for the amount required for class {starClass}")
            existsDf = existsDf.head(size)
            existsDf.to_csv(f"data/raw/new_star_class_data/{starClass}_final_master_data.csv")
            logging.info(f"Missing Data Download Completed for class {starClass}")
        else:
            count = int(size) - int(existsLen)
            logging.info(f"Need to download additonal for the amount required for class {starClass}")
            logging.info(f"Downloading {count} objects for class {starClass}")
            nonExistsDf = nonExistsDf.head(count)
            for index, row in nonExistsDf.iterrows():
                logging.info(f"Attempting to download object: {row['objid']} Index: {index}")
                row['fitsFilePath'] = downloadMissingData(row['objid'], row['plate'], row['mjd'], row['fiberid'])
                existDfNewRows.append(row)
            logging.info(f"Downloading {count} objects for class {starClass} completed")

            existDfNewRowsDF = pd.DataFrame(existDfNewRows)
            existsDf = pd.concat([existsDf, existDfNewRowsDF], ignore_index=True)
            existsDf.to_csv(f"data/raw/new_star_class_data/{starClass}_final_master_data.csv")
            logging.info(f"Missing Data Download Completed for class {starClass}")

        
        
    