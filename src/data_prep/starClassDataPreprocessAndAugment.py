import time
import numpy as np
import requests as reqObj
import pandas as pd
import logging
from datetime import datetime
import os
from scipy.signal import savgol_filter
from astropy.io import fits
from sklearn.preprocessing import MinMaxScaler

from configs.sdssApiEndpoints import SAS_SPEC_FITS_FETCH_BASE, SDSS_OBJ_SQL_SEARCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"
os.makedirs(log_directory, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"new_star_data_preprocess_{timestamp}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

# Yay finally we have all 7 classes
starClasses = ['O', 'B', 'A', 'F', 'G', 'K', 'M']
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
    sasFitsFetchUrl = f"{SAS_SPEC_FITS_FETCH_BASE}plateid={plate}&mjd={mjd}&fiberid={fiberid}"
    fitsDownloader(sasFitsFetchUrl, finalRawFilePath)
    logging.info(f"Completed object download: {objid}")
    logging.info("Initiating Sleep for 0.5 secs to preserve api rate limits")
    time.sleep(0.5)
    return finalRawFilePath

def fitsExtractor(filepath, objid, index):
    logging.info(f"Initiating Fits Extraction for {objid}")
    try:
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            starData = pd.DataFrame({'wavelength': 10 ** data['loglam'], 'flux': data['flux']})
            csv_file_path = f'data/raw/fits_files/spectral_data_extract/star/{objid}_spectra.csv'
            starData.to_csv(csv_file_path, index=False)
            logging.info(f"Done Fits Extraction for {objid}")
            return csv_file_path
        
    except Exception as e:
        logging.error(f"Incomplete or Corrupt File found at index #{index}. Filepath: {filepath} Object ID: {objid}")
        handleCorruptFiles(filepath, objid, index)
        return fitsExtractor(filepath, objid, index)

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

def smoothFlux(csvInputPath, csvOutputPath):
    logging.info(f"Input CSV Path: {csvInputPath}")
    logging.info(f"Output CSV Path: {csvOutputPath}")
    df = pd.read_csv(csvInputPath)
    windowLen = min(51, len(df['flux']) // 2 * 2 + 1)
    df['smoothed_flux'] = savgol_filter(df['flux'], window_length=windowLen, polyorder=3)
    df.to_csv(csvOutputPath, index=False)
    logging.info(f"Smoothed data saved to {csvOutputPath}")

def augmentSpectraData(csvInPath, csvOutPath, objId, starClass):
    spectrumDf = pd.read_csv(csvInPath)
    spectrumDf.sort_values('wavelength', inplace=True)
    wavelengthMin = spectrumDf['wavelength'].min()
    wavelengthMax = spectrumDf['wavelength'].max()
    smoothedFluxMin = spectrumDf['smoothed_flux'].min()
    smoothedFluxMax = spectrumDf['smoothed_flux'].max()
    lambdaMax = spectrumDf.loc[spectrumDf['smoothed_flux'].idxmax(), 'wavelength']
    lambdaMaxMeters = lambdaMax * 1e-10
    outFilePath = f"{csvOutPath}/{objId}_spectra_aug.csv"
    spectrumDf.to_csv(outFilePath, index=False)
    logging.info(f"Written Processed File {objId}_spectra_aug.csv to path: {outFilePath}")
    return (outFilePath, lambdaMax, lambdaMaxMeters, starClass, wavelengthMin, wavelengthMax, smoothedFluxMin, smoothedFluxMax)

def normalizeCsvData(starData, outfilePath, objId):
    logging.info(" ")
    logging.info(f"Initiating Normalizing data for {objId}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    
    normalized_wavelength = scaler.fit_transform(starData[['wavelength']])
    normalized_smoothed_flux = scaler.fit_transform(starData[['smoothed_flux']])
    starData['normalized_wavelength'] = normalized_wavelength
    starData['normalized_smoothed_flux'] = normalized_smoothed_flux

    colSize = colSize = starData['normalized_wavelength'].size
    logging.info(f"Saving Normalized Data to File at path: {outfilePath}")
    starData.to_csv(outfilePath, index=False)
    logging.info(f"Completed Normalizing data for {objId}")
    logging.info(f"Normalized Data File Saved at path: {outfilePath}")
    logging.info(" ")

    return colSize

def padDataFiles(masterData, max_col_size):
    for index, row in masterData.iterrows():
        normalizedFilePath = row['normalizedAndPaddedDataPath']
        starData = pd.read_csv(normalizedFilePath)
        current_size = starData.shape[0]
        padding_size = max_col_size - current_size
        
        if padding_size > 0:
            # Creating a DataFrame of zeros with the same number of columns
            padding = pd.DataFrame(np.zeros((padding_size, starData.shape[1])), columns=starData.columns)
            # Appending the padding to the original data
            starData = pd.concat([starData, padding], ignore_index=True)
            # Saving the padded data back to the file
            starData.to_csv(normalizedFilePath, index=False)
        logging.info(f"Padded {normalizedFilePath} to {max_col_size} rows at index {index}.")


def normalizePhotometricData():
    # Define the paths to the master data files
    masterFiles = [
        'data/augmentation/modelTrainingData/masterFileData/O_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/B_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/A_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/F_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/G_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/K_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/M_class_master_data_stage_4.csv'
    ]
    
    # Columns to scale
    colsToScale = ['u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z', 'effectiveTemperature']
    
    # Load and concatenate the DataFrames to fit the scaler
    logging.info("Loading and concatenating master data files...")
    masterData = pd.concat((pd.read_csv(f) for f in masterFiles), ignore_index=True)
    
    # Initialize and fit the MinMaxScaler
    logging.info("Fitting the MinMaxScaler to the combined data...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(masterData[colsToScale])
    
    # Apply the scaler to each file independently and save the results
    for filePath in masterFiles:
        logging.info(f"Scaling data in {filePath}...")
        data = pd.read_csv(filePath)
        data[colsToScale] = scaler.transform(data[colsToScale])
        data.to_csv(filePath, index=False)
        logging.info(f"Saved scaled data to {filePath}.")

    logging.info("Normalization completed for all master files.")



# ALL STAGES ENTRY POINTS BELOW

# Stage: 1
# Extract Fits Files Wavelength and Flux Data and save to independent CSV
# Write Stage 1 Master File with extraction file paths added
def preprocessStarClassData(starClass, starDataMasterFile):
    logging.info(f"Starting Stage 1 Preprocess for Star Class {starClass}")
    stageOneProcess = []
    colsExist = ['objid','plate','mjd','fiberid', 'u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z', 'effectiveTemperature', 'fitsFilePath', 'fitsExtractCSVInitPath']
    starClassData = pd.read_csv(starDataMasterFile)
    for index, row in starClassData.iterrows():
        objid = row['objid']
        logging.info(f"Checking if filepath exists for object {objid} at index {index}")
        if checkFileExistance(row['fitsFilePath']):
            row['fitsExtractCSVInitPath'] = fitsExtractor(row['fitsFilePath'], objid, index)
            stageOneProcess.append(row)
        else:
            plate, mjd, fiberId = row['plate'], row['mjd'], row['fiberid']
            downloadMissingData(objid, plate, mjd, fiberId)
            row['fitsExtractCSVInitPath'] = fitsExtractor(row['fitsFilePath'], objid, index)
            stageOneProcess.append(row)
    starClassData = pd.DataFrame(stageOneProcess, columns=colsExist)
    starClassData.to_csv(f"data/raw/new_star_class_data/masterFileData/{starClass}_class_master_data_stage_1.csv")
    logging.info(f"Completed Stage 1 Preprocess for Star Class {starClass}")

# Stage: 2
# Process Stage 1 Files and smoothen the flux data and save to another independent CSV
# Write Stage 2 Master File with the smoothened file paths added
def smoothenStarFluxData(starClass, stageOneMasterFilepath):
    logging.info(f"Starting Stage 2 Preprocess for Star Class {starClass}")
    stageTwoProcess = []
    colsExist = ['objid','plate','mjd','fiberid', 'u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z', 'effectiveTemperature', 'fitsFilePath', 'fitsExtractCSVInitPath', 'denoisedCSVDataPath']
    starClassData = pd.read_csv(stageOneMasterFilepath)
    for index, row in starClassData.iterrows():
        objid = row['objid']
        csvIn = row['fitsExtractCSVInitPath']
        csvOut = f"data/processed/newStarClassData/{objid}_spectra_denoised.csv"
        logging.info(f"Applying Denoise Filters for flux for {objid} at index {index}")
        smoothFlux(csvIn, csvOut)
        row['denoisedCSVDataPath'] = csvOut
        stageTwoProcess.append(row)
    starClassData = pd.DataFrame(stageTwoProcess, columns=colsExist)
    starClassData.to_csv(f"data/processed/masterFileData/{starClass}_class_master_data_stage_2.csv")
    logging.info(f"Completed Stage 2 Preprocess for Star Class {starClass}")

# Stage: 3
# Process Stage 2 Files and add more features data and save to another independent CSV
# Write Stage 3 Master File with the augmented file paths
def augmentStarClassData(starClass, stageTwoMasterFilepath):
    logging.info(f"Starting Stage 3 Preprocess for Star Class {starClass}")
    stageThreeProcess = []
    colsExist = ['objid','plate','mjd','fiberid', 'u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z', 'effectiveTemperature', 'fitsFilePath', 'fitsExtractCSVInitPath', 'denoisedCSVDataPath', 'augmentedCSVDataPath', 'lambdaMax', 'lambdaMaxMeters', 'starClass', 'wavelengthMin', 'wavelengthMax', 'smoothedFluxMin', 'smoothedFluxMax']
    starClassData = pd.read_csv(stageTwoMasterFilepath)
    
    for index, row in starClassData.iterrows():
        objid = row['objid']
        csvIn = row['denoisedCSVDataPath']
        csvOut = f"data/augmentation/newStarClassData/{starClass}"
        os.makedirs(csvOut, exist_ok=True)
        logging.info(f"Applying Augmentation for object {objid} at index {index}")
        row['augmentedCSVDataPath'], row['lambdaMax'], row['lambdaMaxMeters'], row['starClass'], row['wavelengthMin'], row['wavelengthMax'], row['smoothedFluxMin'], row['smoothedFluxMax'] = augmentSpectraData(csvIn, csvOut, objid, starClass)
        stageThreeProcess.append(row)
    starClassData = pd.DataFrame(stageThreeProcess, columns=colsExist)
    starClassData.to_csv(f"data/augmentation/masterFileData/{starClass}_class_master_data_stage_3.csv")
    logging.info(f"Completed Stage 3 Preprocess for Star Class {starClass}")

# Stage: 4
# Process Stage 3 Files and normalize the flux and wavelength data and save to another independent CSV
# Write Stage 4 Master File with the normalized file paths
def normalizeStarClassData(starClass, stageThreeMasterFilepath):
    logging.info(f"Starting Stage 4 Preprocess for Star Class {starClass}")
    stageFourProcess = []
    colsExist = ['objid','plate','mjd','fiberid', 'u', 'g', 'r', 'i', 'z', 'u_g', 'g_r', 'r_i', 'i_z', 'effectiveTemperature', 'fitsFilePath', 'fitsExtractCSVInitPath', 'denoisedCSVDataPath', 'augmentedCSVDataPath', 'normalizedAndPaddedDataPath', 'columnSize', 'lambdaMax', 'lambdaMaxMeters', 'starClass', 'wavelengthMin', 'wavelengthMax', 'smoothedFluxMin', 'smoothedFluxMax']
    starClassData = pd.read_csv(stageThreeMasterFilepath)
    
    for index, row in starClassData.iterrows():
        objid = row['objid']
        csvIn = row['augmentedCSVDataPath']
        csvOutDir = f"data/augmentation/modelTrainingData/{starClass}"
        csvOut = f"data/augmentation/modelTrainingData/{starClass}/{row['objid']}_normalized.csv"
        os.makedirs(csvOutDir, exist_ok=True)
        logging.info(f"Applying Normalization for object {objid} at index {index}")

        indStarData = pd.read_csv(csvIn)
        row['normalizedAndPaddedDataPath'] = csvOut
        row['columnSize'] = normalizeCsvData(indStarData, csvOut, objid)
        stageFourProcess.append(row)
    
    maxColSize = max(stageFourProcess, key=lambda x:x['columnSize'])['columnSize']
    starClassData = pd.DataFrame(stageFourProcess, columns=colsExist)
    starClassData.to_csv(f"data/augmentation/modelTrainingData/masterFileData/{starClass}_class_master_data_stage_4.csv")
    logging.info(f"Completed Stage 4 Preprocess for Star Class {starClass}")
    return maxColSize
    
if __name__ == "__main__":
    logging.info("Initiating Star Data Preprocessing")
    
    maxCols = []

    for starClass in starClasses:
        starDataMasterFile = f"data/raw/new_star_class_data/{starClass}_final_master_data.csv"
        starDataStage1 = f"data/raw/new_star_class_data/masterFileData/{starClass}_class_master_data_stage_1.csv"
        starDataStage2 = f"data/processed/masterFileData/{starClass}_class_master_data_stage_2.csv"
        starDataStage3 = f"data/augmentation/masterFileData/{starClass}_class_master_data_stage_3.csv"

        # Stage: 1
        # Extract Fits Files Wavelength and Flux Data and save to independent CSV
        # Write Stage 1 Master File with extraction file paths added
        logging.info(f"Initiating Stage 1 Processing for Class: {starClass}")
        preprocessStarClassData(starClass, starDataMasterFile)
        logging.info(f"Stage 1 Processing Completed for Class : {starClass}")

        # Stage: 2
        # Process Stage 1 Files and smoothen the flux data and save to another independent CSV
        # Write Stage 2 Master File with the smoothened file paths added
        logging.info(f"Initiating Stage 2 Processing for Class: {starClass}")
        smoothenStarFluxData(starClass, starDataStage1)
        logging.info(f"Stage 2 Processing Completed for Class : {starClass}")        

        # Stage: 3
        # Process Stage 2 Files and add more features data and save to another independent CSV
        # Write Stage 3 Master File with the augmented file paths
        logging.info(f"Initiating Stage 3 Processing for Class: {starClass}")
        augmentStarClassData(starClass, starDataStage2)
        logging.info(f"Stage 3 Processing Completed for Class : {starClass}")

        # Stage: 4
        # Process Stage 3 Files and normalize the flux and wavelength data and save to another independent CSV
        # Write Stage 4 Master File with the normalized file paths
        logging.info(f"Initiating Stage 4 Processing for Class: {starClass}")
        curMaxCol = normalizeStarClassData(starClass, starDataStage3)
        maxCols.append(curMaxCol)
        logging.info(f"Stage 4 Processing Completed for Class : {starClass}")

    # Stage: 5
    # Process Stage 4 Files and prep the data by padding the files to a fixed length determined by the max of max cols found in stage 4 and update same CSV
    for starClass in starClasses:
        logging.info(f"Initiating Stage 5 Processing for Class: {starClass}")
        starDataStage4 = f"data/augmentation/modelTrainingData/masterFileData/{starClass}_class_master_data_stage_4.csv"
        padLengthMax = max(maxCols)
        masterData = pd.read_csv(starDataStage4)
        padDataFiles(masterData, padLengthMax)
        logging.info(f"Stage 5 Processing Completed for Class : {starClass}")

    # Stage: 6
    # Process Stage 4 Files and preprocess the photometric data by normalizing the combined values
    normalizePhotometricData()
    
    logging.info("Completed Star Data Preprocessing")