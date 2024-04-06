import requests as reqObj
import pandas as pd
import logging
from datetime import datetime
import os
import time
import threading

from configs.sdssApiEndpoints import SDSS_IMAGE_CUTOUT_BASE, SAS_SPEC_FITS_FETCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_fetch_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"astrometric_data_download_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

def loadAndExtractData(dataFilePath):
    colsToLoad = ['objid', 'specobjid', 'ra', 'dec', 'run2d', 'plate', 'mjd', 'fiberid']
    # Load the data from the CSV file, filtering to only include your specified columns
    df = pd.read_csv(dataFilePath, usecols=colsToLoad, comment='#')
    return df

def fileWriter(response, filePath):
    logging.info(f"Writing {filePath} ....")
    with open(filePath, 'wb') as file:
        file.write(response.content)
        logging.info(f"Written {filePath} ....\n")

def specMetaFetch(astroClass, fileName, imageFilterOptions, photoEnable, sasFetchEnable):
    
    astroClassFileDir = {"GALAXY": "data/raw/csv_extract/galaxy/", "STAR": "data/raw/csv_extract/star/", "QSO": "data/raw/csv_extract/qso/"}
    classPath = astroClassFileDir.get(astroClass)
    astroDataFilePath = classPath + fileName

    specClassFileDir = {"GALAXY": "data/raw/fits_files/spectral_fits/galaxy/", "STAR": "data/raw/fits_files/spectral_fits/star/", "QSO": "data/raw/fits_files/spectral_fits/qso/"}
    specClassPath = specClassFileDir.get(astroClass)

    scale = 0.25
    height = 128
    width = 128
    
    baseImagePathTemplate = "data/raw/image_extracts/astroImages/{filterType}Images/{option}Filter/{astroClass}/"

    astroRawDataFrame = loadAndExtractData(astroDataFilePath)

    i = 0

    # Load only 25000 samples per class
    for row in astroRawDataFrame.iloc[:25000].itertuples(index=False):
        # Assign each field to a variable
        objid = row.objid
        specobjid = row.specobjid
        ra = row.ra
        dec = row.dec
        run2d = row.run2d
        plate = row.plate
        mjd = row.mjd
        fiberid = row.fiberid
        i += 1
        logging.info(f"At Index: {i}")
    
        if(sasFetchEnable):
            # Gets Spectra from Science Archive Server and saves the corresponding class fits file
            sasFitsFetchUrl = SAS_SPEC_FITS_FETCH_BASE + f"plateid={plate}&mjd={mjd}&fiberid={fiberid}"
            sasFitsDataResp = reqObj.get(sasFitsFetchUrl)
            fileWriter(sasFitsDataResp, f"{specClassPath}{objid}_spec.fits")
            logging.info("Initiating Sleep for 3 sec for api server rate limitations")
            time.sleep(3)

        if(photoEnable):
            # Fetch Images from SDSS Server
            if not imageFilterOptions:
                # Handle unfiltered images
                filterPath = baseImagePathTemplate.format(filterType="unFiltered", option="unFiltered", astroClass=astroClass.lower())
                imagePath = f"{filterPath}{objid}.png"
                sdssFetchUrl = f"{SDSS_IMAGE_CUTOUT_BASE}ra={ra}&dec={dec}&scale={scale}&height={height}&width={width}"
                sdssImageDataResp = reqObj.get(sdssFetchUrl)
                os.makedirs(os.path.dirname(imagePath), exist_ok=True)  # Ensure directory exists
                fileWriter(sdssImageDataResp, imagePath)
                logging.info("Initiating Sleep for 3 sec for api server rate limitations")
                time.sleep(3)
            else:
                # Handle filtered images
                for option in imageFilterOptions:
                    filterPath = baseImagePathTemplate.format(filterType="filtered", option=option, astroClass=astroClass.lower())
                    imagePath = f"{filterPath}{objid}_{option}.png"
                    sdssFetchUrl = f"{SDSS_IMAGE_CUTOUT_BASE}ra={ra}&dec={dec}&scale={scale}&height={height}&width={width}&opt='{option}'" 
                    sdssImageDataResp = reqObj.get(sdssFetchUrl)
                    os.makedirs(os.path.dirname(imagePath), exist_ok=True)  # Ensure directory exists
                    fileWriter(sdssImageDataResp, imagePath)
                    logging.info("Initiating Sleep for 3 sec for api server rate limitations")
                    time.sleep(3)

def automateFetching(classes, options):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Fetching Astrometric data initiated.....{timestamp}")

    threads = []  # List to keep track of threads

    for i, astroClass in enumerate(classes):
        try:
            threadName = f"Thread-{astroClass}"  # Name the thread after the astroClass for clarity
            logging.info(f"Initializing fetching for: {astroClass}")
            photoEnable = True
            sasFetchEnable = False
            # Create a Thread for each class, giving it a meaningful name
            thread = threading.Thread(target=specMetaFetch, args=(astroClass, "astro_data_batch_2.csv", options, photoEnable, sasFetchEnable), name=threadName)
            threads.append(thread)
            thread.start()
        except Exception as e:
            logging.exception("Failed to initialize thread for fetching Spectra data.....")

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Fetching Astrometric data completed.....{timestamp}")

if __name__ == "__main__":
    classList = ["GALAXY", "STAR", "QSO"]
    options = [] #'''GT','I','OBFQ'''
    automateFetching(classList, options)