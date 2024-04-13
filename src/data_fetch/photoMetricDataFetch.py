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
log_filename = f"photometric_data_download_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def initDataSizeFetch(astroClass):
    initSqlGetResultCount = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT COUNT(*) FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE s.class = '{}'&format=json".format(astroClass)
    sizeResp = reqObj.get(initSqlGetResultCount)
    if(sizeResp.status_code == 200):
        data = sizeResp.json()
        initDataSize = data[0]['Rows'][0]['Column1']
    
    return initDataSize

def remDataSizeFetch(lastObjId, astroClass):
    sqlGetResultCount = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT COUNT(*) FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE p.objid > " + str(lastObjId) + " AND s.class = '{}'&format=json".format(astroClass)
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

def photometricDataFetch(astroClass):

    iter = 1
    size = initDataSizeFetch(astroClass)

    photoMetricDataPathDir = {"GALAXY": "data/raw/photometricDataExtract/galaxy/", "STAR": "data/raw/photometricDataExtract/star/", "QSO": "data/raw/photometricDataExtract/qso/"}
    photoMetricClassPath = photoMetricDataPathDir.get(astroClass)
    photoMetricFilePath = photoMetricClassPath + f'photometric_data_batch_{astroClass}_{iter}.csv'
    logging.info(photoMetricFilePath)
    initFetchQuery = f"cmd=SELECT p.objid,s.specobjid, s.class, p.ra,p.dec,p.u,p.g,p.r,p.i,p.z, s.plate, s.mjd, s.fiberid, elodieTEff FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE s.class = '{astroClass}' ORDER BY p.objid"
    initFetchUrl = SDSS_OBJ_SQL_SEARCH_BASE + f"{initFetchQuery}&format=csv"
    
    initResponse = reqObj.get(initFetchUrl)
    multiPartFileWriter(initResponse, photoMetricFilePath)
    lastObjectID = getLastObjId(photoMetricFilePath)

    logging.info("Size: {}\tLast Object ID: **NA**\tFile Path: {}".format(size, photoMetricFilePath))    

    while(size != 0):
        iter += 1
        photoMetricFilePath = photoMetricClassPath + f'photometric_data_batch_{astroClass}_{iter}.csv'
        getphotoMetricBatched = SDSS_OBJ_SQL_SEARCH_BASE + f"cmd=SELECT p.objid,s.specobjid, s.class, p.ra,p.dec,p.u,p.g,p.r,p.i,p.z, s.plate, s.mjd, s.fiberid, s.elodieTEff FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE p.objid > {lastObjectID} AND s.class = '{astroClass}' ORDER BY p.objid&format=csv"
        batchedDataResponse = reqObj.get(getphotoMetricBatched)
        multiPartFileWriter(batchedDataResponse, photoMetricFilePath)
        lastObjectID = getLastObjId(photoMetricFilePath)
        size = remDataSizeFetch(lastObjectID, astroClass)
        logging.info("Size: {}\tLast Object ID: {}\tFile Path: {}".format(size, lastObjectID, photoMetricFilePath))


def automateFetching(classes):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Fetching PhotoMetric data initiated.....{timestamp}")

    for astroClass in classes:
        try:
            logging.info(f"Fetching PhotoMetric data for: {astroClass}")
            photometricDataFetch(astroClass)
            logging.info(f"Fetching PhotoMetric data done for: {astroClass}")
        except Exception as e:
            logging.exception(f"Failed to fetch Spectra data.....")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Fetching PhotoMetric data completed.....{timestamp}")

if __name__ == "__main__":
    classList = ["STAR"]
    automateFetching(classList)