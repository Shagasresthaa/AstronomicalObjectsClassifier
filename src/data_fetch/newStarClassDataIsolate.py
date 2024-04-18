import pandas as pd
import logging
from datetime import datetime
import os

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"
os.makedirs(log_directory, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_data_isolator_{timestamp}.log"
log_filepath = os.path.join(log_directory, log_filename)

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s', filemode='w')

def classifStarByTemperature(temp):
    """
    Classifies stars based on their temperature.
    
    Parameters:
    - temp: Temperature of the star in Kelvin.
    
    Returns:
    - The spectral class of the star as a string.
    """
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

def find_unprocessed_entries(master_data_path, photometric_data_path1, photometric_data_path2):
    master_df = pd.read_csv(master_data_path)
    photometric_df1 = pd.read_csv(photometric_data_path1, comment='#')
    photometric_df2 = pd.read_csv(photometric_data_path2, comment='#')

    logging.info(photometric_df1.head())

    # Combine and clean the data
    combined_df = pd.concat([photometric_df1, photometric_df2])
    logging.info("Combining both Star Data CSV Files")
    logging.info(combined_df.head())
    combined_df_cleaned = combined_df[~combined_df['objid'].isin(master_df['ObjID'])]

    # Apply the classification function
    combined_df_cleaned['spectral_class'] = combined_df_cleaned['elodieTEff'].apply(classifStarByTemperature)

    # logging.info the counts of all star classes
    spectral_class_counts = combined_df_cleaned['spectral_class'].value_counts()
    logging.info("Spectral Class Demography for Combined Dataframe independent of Master File Datapoints")
    logging.info(spectral_class_counts)

    master_spectral_class_counts = master_df['StarClass'].value_counts()
    logging.info("Spectral Class Demography for Master File Datapoints")
    logging.info(master_spectral_class_counts)
    combined_df_cleaned.to_csv("data/starSpectraAdditionalRaw.csv")


# Load the two CSV files into dataframes
master_data_path = 'data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv'  
photometric_data_path1 = 'data/raw/photometricDataExtract/star/photometric_data_batch_STAR_1.csv'  
photometric_data_path2 = 'data/raw/photometricDataExtract/star/photometric_data_batch_STAR_2.csv'  

find_unprocessed_entries(master_data_path, photometric_data_path1, photometric_data_path2)
