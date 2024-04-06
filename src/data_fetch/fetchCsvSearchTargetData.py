import requests as rqObj
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
log_filename = f"astro_data_fetch_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def initDataSizeFetch(astroClass):
    initSqlGetResultCount = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT COUNT(*) FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE s.class = '{}'&format=json".format(astroClass)
    sizeResp = rqObj.get(initSqlGetResultCount)
    if(sizeResp.status_code == 200):
        data = sizeResp.json()
        initDataSize = data[0]['Rows'][0]['Column1']
    
    return initDataSize

def remDataSizeFetch(lastObjId, astroClass):
    sqlGetResultCount = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT COUNT(*) FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE p.objid > " + str(lastObjId) + " AND s.class = '{}'&format=json".format(astroClass)
    sizeResp = rqObj.get(sqlGetResultCount)
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

def fetchCsvMultiBatchData(astroClass):
    iter = 1
    size = initDataSizeFetch(astroClass)

    astroClassFileDir = {"GALAXY": "data/raw/csv_extract/galaxy/", "STAR": "data/raw/csv_extract/star/", "QSO": "data/raw/csv_extract/qso/"}
    classPath = astroClassFileDir.get(astroClass)
    astroDataFilePath = classPath + 'astro_data_batch_' + str(iter) + '.csv'

    logging.info("Size: {}\tLast Object ID: **NA**\tFile Path: {}".format(size, astroDataFilePath))
    
    getAstroDataInit = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT p.objid, s.specobjid, s.class, s.run2d, s.plate, s.mjd, s.fiberid, p.ra as ra, p.dec as dec, p.run AS r, p.camcol AS c, p.field as field FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE s.class = '{}' ORDER BY p.objid&format=csv".format(astroClass)
    initResponse = rqObj.get(getAstroDataInit)
    multiPartFileWriter(initResponse, astroDataFilePath)
    lastObjectID = getLastObjId(astroDataFilePath)

    while(size != 0):
        iter += 1
        astroDataFilePath = classPath + 'astro_data_batch_' + str(iter) + '.csv'
        getAstroDataBatched = SDSS_OBJ_SQL_SEARCH_BASE + "cmd=SELECT p.objid, s.specobjid, s.class, s.run2d, s.plate, s.mjd, s.fiberid, p.ra as ra, p.dec as dec, p.run AS r, p.camcol AS c, p.field as field FROM PhotoObj AS p JOIN SpecObj AS s ON s.bestobjid = p.objid WHERE p.objid > {} AND s.class = '{}' ORDER BY p.objid&format=csv".format(lastObjectID, astroClass)
        batchedDataResponse = rqObj.get(getAstroDataBatched)
        multiPartFileWriter(batchedDataResponse, astroDataFilePath)
        lastObjectID = getLastObjId(astroDataFilePath)
        size = remDataSizeFetch(lastObjectID, astroClass)
        logging.info("Size: {}\tLast Object ID: {}\tFile Path: {}".format(size, lastObjectID, astroDataFilePath))

def automateFetching(classes):
    for astroClass in classes:
        try:
            logging.info(f"Fetching data for: {astroClass}")
            fetchCsvMultiBatchData(astroClass)
        except Exception as e:
            logging.exception(f"Failed to fetch data for {astroClass}")

if __name__ == "__main__":
    classList = ["GALAXY", "STAR", "QSO"]
    automateFetching(classList)