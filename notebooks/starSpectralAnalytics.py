from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits

# Load the CSV file into a DataFrame
df = pd.read_csv('data/augmentation/modelData/starNormalizedData/M/1237645942904389833_normalized.csv')  # Update with your CSV file path
    
    

# Assuming your CSV has columns named 'wavelength' and 'flux'
wavelength = df['normalized_wavelength']
flux = df['flux']
smoothed_flux = df['normalized_smoothed_flux']

# Plotting
plt.figure(figsize=(10, 6))  # Optional: Adjusts the figure size
plt.plot(wavelength, smoothed_flux, label='Flux vs. Wavelength')
plt.xlabel('Wavelength')
plt.ylabel('Flux')
plt.title('Flux as a Function of Wavelength')
plt.legend()
plt.grid(True)
# Save the plot to a file
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
plt.savefig(f'notebooks/spectral_plots/stars_sample_flux_smoothened_denoised_1237645942906028124_{timestamp}.png')
plt.show()