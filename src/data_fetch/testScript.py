import pandas as pd
from sklearn.utils import shuffle


def generateCombinedMasterFile():
    # Load and concatenate the DataFrames
    master_files = [
        'data/augmentation/modelTrainingData/masterFileData/O_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/B_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/A_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/F_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/G_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/K_class_master_data_stage_4.csv',
        'data/augmentation/modelTrainingData/masterFileData/M_class_master_data_stage_4.csv'
    ]
    num_samples_per_class = 4000
    masterDataFiles = pd.concat((pd.read_csv(f, nrows=num_samples_per_class) for f in master_files), ignore_index=True)

    # Shuffle the combined DataFrame
    masterDataFiles = shuffle(masterDataFiles, random_state=42)
    masterDataFiles.reset_index(drop=True, inplace=True)
    masterDataFiles = shuffle(masterDataFiles, random_state=72)
    # Save or further process the shuffled DataFrame
    masterDataFiles.to_csv('data/augmentation/modelTrainingData/combined_master_star_classes_shuffled.csv', index=False)

generateCombinedMasterFile()