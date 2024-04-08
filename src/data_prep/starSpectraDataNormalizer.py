import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os
from sklearn.preprocessing import MinMaxScaler

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_spectra_normalizer_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

masterFilePath = "data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv"

def normalizeCsvData(starData, outfilePath, outFileDir, objId):
    logging.info(" ")
    logging.info(f"Initiating Normalizing data for {objId}")

    scaler = MinMaxScaler(feature_range=(0, 1))
    
    normalized_wavelength = scaler.fit_transform(starData[['wavelength']])
    normalized_smoothed_flux = scaler.fit_transform(starData[['smoothed_flux']])
    starData['normalized_wavelength'] = normalized_wavelength
    starData['normalized_smoothed_flux'] = normalized_smoothed_flux
    colSize = starData.iloc[:, 0].size
    os.makedirs(outFileDir, exist_ok=True)
    logging.info(f"Saving Normalized Data to File at path: {outfilePath}")
    starData.to_csv(outfilePath, index=False)
    logging.info(f"Completed Normalizing data for {objId}")
    logging.info(f"Normalized Data File Saved at path: {outfilePath}")
    logging.info(" ")

    return colSize

def padDataFiles(masterData, max_col_size):
    for index, row in masterData.iterrows():
        normalizedFilePath = row['NormalizedFilePath']
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
        logging.info(f"Padded {normalizedFilePath} to {max_col_size} rows.")

def normalizeEverything():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Normalizing Spectral data from spectra csv files initiated.....{timestamp}")
    logging.info(" ")

    masterData = pd.read_csv(masterFilePath, comment='#')
    masterDataList = []
    columns=['ObjID', 'OutputFilePath', 'NormalizedFilePath', 'ColumnSize', 'TemperatureKelvin', 'MaxWavelengthAngstroms', 'MaxWaveLengthMeters', 'StarClass', 'WavelengthMin', 'WavelengthMax', 'SmoothedFluxMin', 'SmoothedFluxMax']
    

    try:        
        for index, row in masterData.iterrows():
            # Path to your original data CSV file
            outFilePath = f"data/augmentation/modelData/starNormalizedData/{row['StarClass']}/{row['ObjID']}_normalized.csv"
            outFileDir = f"data/augmentation/modelData/starNormalizedData/{row['StarClass']}/"
            starCsv = row['OutputFilePath']
            starData = pd.read_csv(starCsv)
            colSize = normalizeCsvData(starData, outFilePath, outFileDir, row['ObjID'])

            mastrerDataRow = {
                'ObjID': row['ObjID'],
                'OutputFilePath': row['OutputFilePath'], 
                'ColumnSize': colSize, 
                'NormalizedFilePath': outFilePath, 
                'TemperatureKelvin': row['TemperatureKelvin'], 
                'MaxWavelengthAngstroms': row['MaxWavelengthAngstroms'], 
                'MaxWaveLengthMeters': row['MaxWaveLengthMeters'], 
                'StarClass': row['StarClass'], 
                'WavelengthMin': row['WavelengthMin'], 
                'WavelengthMax': row['WavelengthMax'], 
                'SmoothedFluxMin': row['SmoothedFluxMin'], 
                'SmoothedFluxMax': row['SmoothedFluxMax']
            }
            masterDataList.append(mastrerDataRow)
        max_col_size = max(masterDataList, key=lambda x:x['ColumnSize'])['ColumnSize']
        masterFile = pd.DataFrame(masterDataList, columns=columns)
        masterFile.to_csv("data/augmentation/modelData/starNormalizedData/masterDataFile.csv")

        # Now pad the files to the max column size
        padDataFiles(masterFile, max_col_size)
    
        logging.info("All files have been padded to the maximum column size.")
            
    except Exception as e:
        logging.exception(f"Failed to Normalize Spectra data.....")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(" ")
    logging.info(f"Normalizing Spectral data from spectra csv files completed.....{timestamp}")

if __name__ == "__main__":
    normalizeEverything()