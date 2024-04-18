import requests as reqObj
import pandas as pd
import logging
from datetime import datetime
import os
import time
import threading

from configs.sdssApiEndpoints import SDSS_OBJ_SQL_SEARCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_fetch_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_class_data_download_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

starClasses = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

def initDataSizeFetch(starClass):
    initSqlGetResultCount = f"{SDSS_OBJ_SQL_SEARCH_BASE}cmd=SELECT COUNT(*) from specobj WHERE class = 'STAR' and subclass like '{starClass}%' and bestobjid != 0&format=json"
    sizeResp = reqObj.get(initSqlGetResultCount)
    if(sizeResp.status_code == 200):
        data = sizeResp.json()
        initDataSize = data[0]['Rows'][0]['Column1']
    
    return initDataSize

def remDataSizeFetch(lastObjId, starClass):
    sqlGetResultCount = f"{SDSS_OBJ_SQL_SEARCH_BASE}cmd=SELECT COUNT(*) from specobj WHERE bestobjid > {lastObjId} AND class = 'STAR' and subclass like '{starClass}%' and bestobjid != 0&format=json"
    sizeResp = reqObj.get(sqlGetResultCount)
    if(sizeResp.status_code == 200):
        data = sizeResp.json()
        dataSize = data[0]['Rows'][0]['Column1']
    
    return dataSize

def multiPartFileWriter(resp, csvFilePath):
    with open(csvFilePath, 'wb') as file:
        file.write(resp.content)

def getLastObjId(lastBatchFile):
    df = pd.read_csv(lastBatchFile, comment='#')
    last_objid = df['objid'].iloc[-1]
    return last_objid

def fetchCsvMultiBatchData(starClass):
    iter = 1
    size = initDataSizeFetch(starClass)
    logging.info(f"Size: {size}")
    rawStarDir = 'data/raw/new_star_class_data/'
    os.makedirs(rawStarDir, exist_ok=True)
    initialDataUrl = f"{SDSS_OBJ_SQL_SEARCH_BASE}cmd=SELECT bestobjid as objid, plate, mjd, fiberid, elodieTEff as effectiveTemperature FROM specobj WHERE class = 'STAR' AND subclass LIKE '{starClass}%' AND bestobjid != 0 ORDER BY bestobjid&format=csv"
    initResponse = reqObj.get(initialDataUrl)
    csvFilePath = f'{rawStarDir}{starClass}_star_class_{iter}.csv'
    multiPartFileWriter(initResponse, csvFilePath)
    lastObjectID = getLastObjId(csvFilePath)
    size = remDataSizeFetch(lastObjectID, starClass)

    while(size != 0):
        iter += 1
        csvFilePath = f'{rawStarDir}{starClass}_star_class_{iter}.csv'
        getAstroDataBatched = f"{SDSS_OBJ_SQL_SEARCH_BASE}cmd=SELECT bestobjid as objid, plate, mjd, fiberid, elodieTEff as effectiveTemperature FROM specobj WHERE bestobjid > {lastObjectID} AND class = 'STAR' and subclass like '{starClass}%' and bestobjid != 0 ORDER BY bestobjid&format=csv"
        batchedDataResponse = reqObj.get(getAstroDataBatched)
        multiPartFileWriter(batchedDataResponse, csvFilePath)
        lastObjectID = getLastObjId(csvFilePath)
        size = remDataSizeFetch(lastObjectID, starClass)
        logging.info("Size: {}\tLast Object ID: {}\tFile Path: {}".format(size, lastObjectID, csvFilePath))

def downloadStarClassData():
    logging.info("Starting to fetch Star Class Data")
    for starClass in starClasses:
        try:
            logging.info(f"Fetching data for: {starClass}")
            fetchCsvMultiBatchData(starClass)
        except Exception as e:
            logging.exception(f"Failed to fetch data for {starClass}")

    logging.info("Completed fetching Star Class Data")
if __name__ == "__main__":
    downloadStarClassData()