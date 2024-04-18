from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the CSV file into a DataFrame
df = pd.read_csv('data/augmentation/modelTrainingData/O/1197132287879676584_normalized.csv')  # Update with your CSV file path
    
df = df.head(3846)    

# Assuming your CSV has columns named 'wavelength' and 'flux'
wavelength = df['normalized_wavelength']
flux = df['flux']
smoothed_flux = df['normalized_smoothed_flux']
wavelength_unit = 'Å'  # Angstroms
flux_unit = 'erg/s/cm²/Å'
# Plotting
plt.figure(figsize=(10, 6))  # Optional: Adjusts the figure size
plt.plot(wavelength, smoothed_flux, label='Smoothed Flux vs. Wavelength')
plt.xlabel(f'Wavelength ({wavelength_unit})')
plt.ylabel(f'Flux ({flux_unit})')
plt.title('Flux as a Function of Wavelength (Normalized and Padded datapoints)')
plt.legend()
plt.grid(True)
# Save the plot to a file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f'notebooks/spectral_plots/class_O_spectra_normalized.png')
plt.show()