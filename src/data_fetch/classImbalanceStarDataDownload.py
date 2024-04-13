import time
import requests as reqObj
import pandas as pd
import logging
from datetime import datetime
import os
from scipy.signal import savgol_filter
from astropy.io import fits

from configs.sdssApiEndpoints import SAS_SPEC_FITS_FETCH_BASE, SDSS_OBJ_SQL_SEARCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_fetch_logs"
os.makedirs(log_directory, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"new_star_data_download_{timestamp}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

def initializeDataFrames():
    masterDf = pd.read_csv('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv')
    raw_df = pd.read_csv('data/starSpectraAdditionalRaw.csv')
    
    # Filter out class 'B' from both DataFrames
    masterDf = masterDf[masterDf['StarClass'] != 'B']
    raw_df = raw_df[raw_df['spectral_class'] != 'B']
    
    addedRowsDf = pd.DataFrame()

    # Processing classes ensuring no class 'B' data is processed
    for spectral_class in masterDf['StarClass'].unique():
        if spectral_class == 'B':  # Skip class 'B'
            continue
        class_count = masterDf[masterDf['StarClass'] == spectral_class].shape[0]
        if class_count > 3000:
            masterDf = masterDf.drop(masterDf[masterDf['StarClass'] == spectral_class][3000:].index)
        elif class_count < 3000:
            rows_needed = 3000 - class_count
            rows_to_add = raw_df[raw_df['spectral_class'] == spectral_class].head(rows_needed)
            addedRowsDf = pd.concat([addedRowsDf, rows_to_add], ignore_index=True)

    masteSpectralClassCounts = masterDf['StarClass'].value_counts()
    logging.info("Master Data File Class Infographic:")
    logging.info(masteSpectralClassCounts)

    spectralClassCounts = addedRowsDf['spectral_class'].value_counts()
    logging.info("Additional Raw Data Class Files Required Infographic:")
    logging.info(spectralClassCounts)

    masterDf.to_csv('data/modified_master.csv', index=False)
    addedRowsDf.to_csv('data/added_rows.csv', index=False)
    return masterDf, addedRowsDf

def fitsExtractor(filepath, objid, index):
    try:
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            starData = pd.DataFrame({'wavelength': 10 ** data['loglam'], 'flux': data['flux']})
            csv_file_path = f'data/raw/fits_files/spectral_data_extract/star/{objid}_spectra.csv'
            starData.to_csv(csv_file_path, index=False)
            return csv_file_path
    except Exception as e:
        logging.error(f"Corrupt File found at index #{index}. Filepath: {filepath} Object ID: {objid}")
        handleCorruptFiles(filepath, objid, index)
        return fitsExtractor(filepath, objid, index)

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

def handleCorruptFiles(filepath, objid, index):
    logging.info(f"Deleting Corrupt File found at index #{index}. Filepath: {filepath} Object ID: {objid}")
    os.remove(filepath)
    plate, mjd, fiberId = getSpecParams(objid)
    sasFitsFetchUrl = f"{SAS_SPEC_FITS_FETCH_BASE}plateid={plate}&mjd={mjd}&fiberid={fiberId}"
    fitsDownloader(sasFitsFetchUrl, filepath)

def getSpecParams(objid):
    sql = f"SELECT plate, mjd, fiberId FROM specObj WHERE bestobjid = '{objid}'"
    sdssFetchUrl = f"{SDSS_OBJ_SQL_SEARCH_BASE}cmd={sql}&format=json"
    specSearchData = reqObj.get(sdssFetchUrl).json()
    return specSearchData[0]['Rows'][0]['plate'], specSearchData[0]['Rows'][0]['mjd'], specSearchData[0]['Rows'][0]['fiberId']

def spectralPointerData(objid):
    plate, mjd, fiberId = getSpecParams(objid)
    filePath = f"data/raw/fits_files/spectral_fits/star/{objid}_spec.fits"
    sasFitsFetchUrl = f"{SAS_SPEC_FITS_FETCH_BASE}plateid={plate}&mjd={mjd}&fiberid={fiberId}"
    # Dont download the same files again and again give the server api a break
    if os.path.exists(filePath):
        logging.info("File Already Exists skipping download")
        return filePath
    else:
        logging.info(f"File doesnt exist downloading file to {filePath}")
        fitsDownloader(sasFitsFetchUrl, filePath)
        return filePath

def smoothFlux(csv_input_path, csv_output_path):
    logging.info(f"Input CSV Path: {csv_input_path}")
    logging.info(f"Output CSV Path: {csv_output_path}")
    df = pd.read_csv(csv_input_path)
    df['smoothed_flux'] = savgol_filter(df['flux'], window_length=51, polyorder=3)
    df.to_csv(csv_output_path, index=False)
    logging.info(f"Smoothed data saved to {csv_output_path}")

def augmentSpectraData(csvInPath, objId, temperature, starClass):
    spectrumDf = pd.read_csv(csvInPath)
    spectrumDf.sort_values('wavelength', inplace=True)
    wavelength_min = spectrumDf['wavelength'].min()
    wavelength_max = spectrumDf['wavelength'].max()
    smoothed_flux_min = spectrumDf['smoothed_flux'].min()
    smoothed_flux_max = spectrumDf['smoothed_flux'].max()
    lambdaMax = spectrumDf.loc[spectrumDf['smoothed_flux'].idxmax(), 'wavelength']
    lambdaMaxMeters = lambdaMax * 1e-10
    csvOutPath = f"data/augmentation/finalProcessedSpectrumData/star/{starClass}"
    os.makedirs(csvOutPath, exist_ok=True)
    outFilePath = f"{csvOutPath}/{objId}_spectra_main.csv"
    spectrumDf.to_csv(outFilePath, index=False)
    logging.info(f"Written Processed File {objId}_spectra_main.csv to path: {outFilePath}")
    return (outFilePath, temperature, lambdaMax, lambdaMaxMeters, starClass,
            wavelength_min, wavelength_max, smoothed_flux_min, smoothed_flux_max)

if __name__ == "__main__":
    masterDf, additionalRowDf = initializeDataFrames()
    classifyResults = []
    for index, row in additionalRowDf.iterrows():
        if row['spectral_class'] == 'B':  # Skip processing for class 'B'
            continue
        logging.info(f"Processing Object {row['objid']} at index {index}")
        specFilePath = spectralPointerData(row['objid'])
        fitsInitCsv = fitsExtractor(specFilePath, row['objid'], index)
        csvOutPath = f"data/processed/starSpectralNoiseReducedData/{row['objid']}_spectra_denoised.csv"
        smoothFlux(fitsInitCsv, csvOutPath)
        result = augmentSpectraData(csvOutPath, row['objid'], row['elodieTEff'], row['spectral_class'])
        classifyResults.append((row['objid'],) + result)

    df_results = pd.DataFrame(classifyResults, columns=['ObjID', 'OutputFilePath', 'TemperatureKelvin', 'MaxWavelengthAngstroms', 'MaxWaveLengthMeters', 'StarClass', 'WavelengthMin', 'WavelengthMax', 'SmoothedFluxMin', 'SmoothedFluxMax'])
    masterDf = pd.concat([masterDf, df_results], ignore_index=True)
    if os.path.exists('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv'):
        os.remove('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv')
    masterDf.to_csv('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv', index=False)
