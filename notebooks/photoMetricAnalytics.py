import pandas as pd
import logging
from datetime import datetime
import os

import matplotlib.pyplot as plt

# Logging setup with timestamped filenames
log_directory = "logs/data_analytics_logs"

# Ensure the log directory exists
os.makedirs(log_directory, exist_ok=True)  
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = f"photometric_data_analytics_{timestamp}.log"
log_filepath = f"{log_directory}/{log_filename}"

logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')

def photometricAnalytics(photometricData):

    # Define a dictionary for class-color mapping
    class_colors = {
        'GALAXY': 'red',
        'STAR': 'green',
        'QSO': 'blue'
    }

    # Create a figure and axis for the plot
    plt.figure(figsize=(15, 10))

    # Set the marker size and opacity (alpha) for the scatter plot
    marker_size = 10
    alpha_value = 0.5  

    core_x_min, core_x_max = -1, 4  # Adjust based on where most data points lie
    core_y_min, core_y_max = -1, 4  # Adjust similarly

    logging.info("Starting Plotting")
    for index, row in photometricData.iterrows():
        color = class_colors.get(row['class'], 'gray')  # Default to 'gray' if class is not found
        # Plot 'g-r' vs 'u-g' for the current row with the dynamically determined color
        plt.scatter(row['u-g'], row['g-r'], color=color, s=marker_size, alpha=alpha_value)

    logging.info("Plotting Completed")
    # After plotting all points, add plot decorations
    # Setting plot limits to the core range
    plt.xlim([core_x_min, core_x_max])
    plt.ylim([core_y_min, core_y_max])
    plt.xlabel('u-g')
    plt.ylabel('g-r')
    plt.title("Photometric Data Color Indices Scatter Plot 'u-g' vs 'g-r'")
    # Creating a custom legend manually
    plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=5) for color in class_colors.values()], labels=class_colors.keys())

    # Save the plot to a file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'notebooks/res_{timestamp}.png')

    # Display the plot
    plt.show()

def automateAnalysis(classes):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Processing PhotoMetric data initiated.....{timestamp}")

    # Combining all classes data
    procPhotoDataPath = "data/processed/photoMetricData/"

    combinationDataframe = []


    for astroClass in classes:
        procFile = procPhotoDataPath + f"combined_photometric_data_{astroClass}.csv"
        df = pd.read_csv(procFile, comment='#')
        combinationDataframe.append(df.iloc[:5000])

    combined_df = pd.concat(combinationDataframe, ignore_index=True)
    combined_df['u-g'] = combined_df['u'] - combined_df['g']
    combined_df['g-r'] = combined_df['g'] - combined_df['r']
    
    x_min, x_max = combined_df['u-g'].min(), combined_df['u-g'].max()
    y_min, y_max = combined_df['g-r'].min(), combined_df['g-r'].max()

    logging.info(f"x range: {x_min} to {x_max}")
    logging.info(f"y range: {y_min} to {y_max}")

    logging.info(f"Mean 'u-g': {combined_df['u-g'].mean()}\tMedian 'u-g': {combined_df['u-g'].median()}")
    logging.info(f"Mean 'g-r': {combined_df['g-r'].mean()}\tMedian 'g-r': {combined_df['g-r'].median()}")

    photometricAnalytics(combined_df)

    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logging.info(f"Processing PhotoMetric data completed.....{timestamp}")

if __name__ == "__main__":
    classList = ["GALAXY", "QSO"]
    automateAnalysis(classList)