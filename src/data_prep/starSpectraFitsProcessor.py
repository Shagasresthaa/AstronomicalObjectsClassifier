import subprocess
from astropy.io import fits
import pandas as pd
import logging
from datetime import datetime
import os
import time
import requests as reqObj

from configs.sdssApiEndpoints import SAS_SPEC_FITS_FETCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_spectra_data_extract_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"


logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def fitsExtractor(filepath, objid, index):

    try:
        # Open the FITS file
        logging.info(f"Now Reading File #{index + 1} with Object ID - {objid} at path: {filepath}")
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            loglam = data['loglam']
            flux = data['flux']
            wavelength = 10 ** loglam
            starData = pd.DataFrame({'wavelength': wavelength, 'flux': flux})
    except Exception as e:
        logging.error(f"Corrupt File found at index #{index}. Filepath: {filepath} Object ID: {objid}")
        handleCorruptFiles(filepath, objid, index)
        logging.error(f"Corrupt File handled and new data downloaded")
        fitsExtractor(filepath, objid, index)
        return
    
    logging.info(f"Read and processed data from file #{index + 1} with Object ID - {objid} at path: {filepath}")
    
    logging.info(" ")
    logging.info("Sample Head of Data:")
    logging.info(starData.head())
    logging.info(" ")

    # Replace 'output_file.csv' with your desired output CSV file name
    csv_file_path = f'data/raw/fits_files/spectral_data_extract/star/{objid}_spectra.csv'
    logging.info(f"Writing processed data from file #{index + 1} with Object ID - {objid} to path: {csv_file_path}")
    # Save the DataFrame to a CSV file
    starData.to_csv(csv_file_path, index=False)
    logging.info(f"Written processed data from file #{index + 1} with Object ID - {objid} to path: {csv_file_path}")

def fitsDownloader(response, filePath):
    logging.info(f"Writing {filePath} ....")
    with open(filePath, 'wb') as file:
        file.write(response.content)
        logging.info(f"Written {filePath} ....\n")

def download_with_wget(url, output_path):
    command = ['wget', '-c', '-O', output_path, url]
    subprocess.run(command)

def handleCorruptFiles(filepath, objid, index):
    logging.info(f"Deleting Corrupt File found at index #{index}. Filepath: {filepath} Object ID: {objid}")
    os.remove(filepath)
    logging.info(f"Retreiving Spectral Parameters for Object ID: {objid}")
    astroStarData = pd.read_csv("data/raw/csv_extract/star/astro_data_batch_1.csv", comment='#', usecols=['objid', 'plate', 'mjd', 'fiberid'])
    targetData = astroStarData[astroStarData['objid'] == objid]
    logging.info(f"Retreived Spectral Parameters for Object ID: {objid}:")
    logging.info(targetData)

    logging.info(f"Re-downloading Spectral Data for Object ID: {objid}:")
    plateid = targetData['plate'].iloc[0]
    mjd = targetData['mjd'].iloc[0]
    fiberid = targetData['fiberid'].iloc[0]
    sasFitsFetchUrl = SAS_SPEC_FITS_FETCH_BASE + f"plateid={plateid}&mjd={mjd}&fiberid={fiberid}"
    sasFitsDataResp = reqObj.get(sasFitsFetchUrl)
    fitsDownloader(sasFitsDataResp, filepath)

def autoFitsExtract():

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Processing Spectral data from FITS files initiated.....{timestamp}")

    starData = pd.read_csv("data/raw/csv_extract/star/astro_data_batch_1.csv", comment='#', usecols=['objid'])
    starData = starData.iloc[:25000]

    try:        
        for index, row in starData.iterrows():
            fitsExtractor(f"data/raw/fits_files/spectral_fits/star/{row['objid']}_spec.fits", row['objid'], index)
    except Exception as e:
        logging.exception(f"Failed to process Spectra data.....")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Processing Spectral data from FITS files completed.....{timestamp}")

if __name__ == "__main__":
    autoFitsExtract()