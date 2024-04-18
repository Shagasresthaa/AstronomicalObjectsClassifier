import requests as rqObj
import pandas as pd
import logging
from datetime import datetime
import os

from configs.sdssApiEndpoints import SDSS_OBJ_SQL_SEARCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"
os.makedirs(log_directory, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"temp_class_corrections_{timestamp}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def classifStarByTemperature(temp):
    if temp >= 30000:
        return 'O'
    elif 10000 <= temp < 30000:
        return 'B'
    elif 7500 <= temp < 10000:
        return 'A'
    elif 6000 <= temp < 7500:
        return 'F'
    elif 5200 <= temp < 6000:
        return 'G'
    elif 3700 <= temp < 5200:
        return 'K'
    elif temp < 3700:
        return 'M'
    else:
        return 'Unknown'

def correctTempFromSDSS(objid):
    url = f"{SDSS_OBJ_SQL_SEARCH_BASE}cmd=SELECT elodieTEff FROM specObj WHERE bestobjid = '{objid}'&format=json"
    try:
        resp = rqObj.get(url)
        resp.raise_for_status()  # Raise an HTTPError on bad requests
        data = resp.json()
        actualTemp = data[0]['Rows'][0]['elodieTEff']
        return actualTemp
    except rqObj.exceptions.HTTPError as e:
        logging.error(f"HTTPError for objid {objid}: {str(e)}")
    except Exception as e:
        logging.error(f"Error for objid {objid}: {str(e)}")
    return None  # Return None if any error occurs

if __name__ == "__main__":
    masterFile = "data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv"
    masterFileData = pd.read_csv(masterFile)

    for index, row in masterFileData.iterrows():
        actualTemp = correctTempFromSDSS(row['ObjID'])
        if actualTemp:
            masterFileData.at[index, 'TemperatureKelvin'] = actualTemp
            logging.info(f"Updated temperature for ObjID {row['ObjID']} to {actualTemp}")

    # Apply the new classification based on temperature and save the updated DataFrame back to the CSV
    masterFileData['newStarClass'] = masterFileData['TemperatureKelvin'].apply(classifStarByTemperature)
    if os.path.exists('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv'):
        os.remove('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv')
    masterFileData.to_csv(masterFile, index=False)
    logging.info("All temperatures updated and master file saved.")
