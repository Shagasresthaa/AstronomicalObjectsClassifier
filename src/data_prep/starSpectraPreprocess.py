import pandas as pd
from scipy.signal import savgol_filter
import pandas as pd
import logging
from datetime import datetime
import os

from configs.sdssApiEndpoints import SAS_SPEC_FITS_FETCH_BASE

# Logging setup with timestamped filenames
log_directory = "logs/data_processing_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"star_spectra_data_denoiser_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"


logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def smooth_flux(csv_input_path, csv_output_path):
    df = pd.read_csv(csv_input_path)
    df['smoothed_flux'] = savgol_filter(df['flux'], window_length=51, polyorder=3)

    # Save the DataFrame with the smoothed flux to a new CSV file
    logging.info(f"Saving Smoothed data to {csv_output_path}")
    df.to_csv(csv_output_path, index=False)
    logging.info(f"Smoothed data saved to {csv_output_path}")



def autoDenoiseStarSpectralData():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Denoising Spectral data from spectra csv files initiated.....{timestamp}")

    starData = pd.read_csv("data/raw/csv_extract/star/astro_data_batch_1.csv", comment='#', usecols=['objid'])
    starData = starData.iloc[:25000]

    try:        
        for index, row in starData.iterrows():
            # Path to your original data CSV file
            csv_input_path = f"data/raw/fits_files/spectral_data_extract/star/{row['objid']}_spectra.csv"
            # Path to save the smoothed data CSV file  
            csv_output_path = f"data/processed/starSpectralNoiseReducedData/{row['objid']}_spectra_denoised.csv"
            smooth_flux(csv_input_path, csv_output_path)
    except Exception as e:
        logging.exception(f"Failed to Denoise Spectra data.....")
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Denoising Spectral data from spectra csv files completed.....{timestamp}")

if __name__ == "__main__":
    autoDenoiseStarSpectralData()