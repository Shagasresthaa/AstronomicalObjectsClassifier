import numpy as np
import pandas as pd
import logging
from datetime import datetime
import requests as rqObj
import os

from configs.sdssApiEndpoints import SDSS_OBJ_SQL_SEARCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_spectra_augmenter_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

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

def getTemperatureFromSDSS(objid):
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

def augmentSpectraDataWithStarClass(csvInPath, objId):
    spectrumDf = pd.read_csv(csvInPath)
    spectrumDf.sort_values('wavelength', inplace=True)

    # Calculate min and max for wavelength and smoothed_flux
    wavelength_min = spectrumDf['wavelength'].min()
    wavelength_max = spectrumDf['wavelength'].max()
    smoothed_flux_min = spectrumDf['smoothed_flux'].min()
    smoothed_flux_max = spectrumDf['smoothed_flux'].max()

    lambdaMax = spectrumDf.loc[spectrumDf['smoothed_flux'].idxmax(), 'wavelength']
    b = 2.897e-3  # Weins Displacement Constant
    lambdaMaxMeters = lambdaMax * 1e-10  # Convert Angstroms to meters

    # No longer used as SDSS already has the most accurate measurements in SpecObj Table
    #temperature = b / lambdaMaxMeters   # Calculate Temperature using Weins Displacement Law

    temperature = getTemperatureFromSDSS(objId)
    starClassExpected = classifStarByTemperature(temperature)
    
    logging.info(f"Estimated Temperature and Class for object {objId}:")
    logging.info(f"Star Class: {starClassExpected}\tTemperature Estimate: {temperature} K")
    
    csvOutPath = f"data/augmentation/finalProcessedSpectrumData/star/{starClassExpected}"
    os.makedirs(csvOutPath, exist_ok=True)
    outFilePath = f"{csvOutPath}/{objId}_spectra_main.csv"
    spectrumDf.to_csv(outFilePath, index=False)

    return (outFilePath, temperature, lambdaMax, lambdaMaxMeters, starClassExpected,
            wavelength_min, wavelength_max, smoothed_flux_min, smoothed_flux_max)


def autoAugmentData():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Augmenting Spectral data from spectra csv files initiated.....{timestamp}")

    starData = pd.read_csv("data/raw/csv_extract/star/astro_data_batch_1.csv", comment='#', usecols=['objid'])
    starData = starData.iloc[:25000]

    classifyResults = []

    try:        
        for index, row in starData.iterrows():
            logging.info(f"Index Counter: {index}")
            csvInPath = f"data/processed/starSpectralNoiseReducedData/{row['objid']}_spectra_denoised.csv"
            result = augmentSpectraDataWithStarClass(csvInPath, row['objid'])
            classifyResults.append((row['objid'],) + result)
        
        columns=['ObjID', 'OutputFilePath', 'TemperatureKelvin', 'MaxWavelengthAngstroms', 'MaxWaveLengthMeters', 'StarClass', 'WavelengthMin', 'WavelengthMax', 'SmoothedFluxMin', 'SmoothedFluxMax']
        df_results = pd.DataFrame(classifyResults, columns=columns)
        df_results.to_csv('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv', index=False)
            
    except Exception as e:
        logging.exception(f"Failed to Augment Spectra data.....")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Augmenting Spectral data from spectra csv files completed.....{timestamp}")

if __name__ == "__main__":
    autoAugmentData()