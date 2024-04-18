import time
import requests as reqObj
import pandas as pd
import logging
from datetime import datetime
import os

from configs.sdssApiEndpoints import SDSS_OBJ_SQL_SEARCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_fetch_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"photometric_additional_data_fetch_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def getPhotometricData(starClass, masterFile):
    masterData = pd.read_csv(masterFile)
    updatedMaster = []
    cols = ['objid', 'plate', 'mjd', 'fiberid', 'u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z','effectiveTemperature', 'fitsFilePath']
    for index, row in masterData.iterrows():
        url = f"{SDSS_OBJ_SQL_SEARCH_BASE}cmd=select objid, u, g, r, i, z from PhotoObjAll where objid = '{row['objid']}'&format=json"
        resp = reqObj.get(url)
        logging.info(resp.json())
        data = resp.json()
        if data[0]['Rows']:
            logging.info(f"Received data from SDSS for object {row['objid']} for {starClass} now extracting data at index {index}")
            data = resp.json()[0]['Rows'][0]
            row['u'], row['g'], row['r'], row['i'], row['z'] = data['u'], data['g'], data['r'], data['i'], data['z']
            row['u_g'] = row['u'] - row['g']
            row['g_r'] = row['g'] - row['r']
            row['r_i'] = row['r'] - row['i']
            row['i_z'] = row['i'] - row['z']
            #logging.info(row['u'], row['g'], row['r'], row['i'], row['z'], row['u_g'], row['g_r'], row['r_i'], row['i_z'])
            updatedMaster.append(row)
            logging.info(f"Successfully Added data from SDSS for object {row['objid']} for {starClass} at index {index}")
            logging.info("Respecting API Limits now sleeping for 1000 ms")
            time.sleep(0.5)
        else:
            logging.warn(f"No data found for object {row['objid']} and setting all photometric indices to 0")
            row['u'], row['g'], row['r'], row['i'], row['z'], row['u_g'], row['g_r'], row['r_i'], row['i_z'] = 0,0,0,0,0,0,0,0,0
            updatedMaster.append(row)

        
        
    newMasterData = pd.DataFrame(updatedMaster, columns=cols)
    newMasterData.to_csv(masterFile)

def automateFetching(classes):
    for starClass in classes:
        logging.info(f"Starting Master File Additions for class {starClass}")
        masterFile = f'data/raw/new_star_class_data/{starClass}_final_master_data.csv'
        getPhotometricData(starClass, masterFile)
        logging.info(f"Completed Master File Additions for class {starClass}")

if __name__ == "__main__":
    classList = ['K', 'M']
    automateFetching(classList)