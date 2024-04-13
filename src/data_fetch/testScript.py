import pandas as pd

classes = ['F', 'K', 'M']

for starClass in classes:
    filePath = f'data/raw/new_star_class_data/{starClass}_existing_master_data.csv'
    starData = pd.read_csv(filePath)
    starData = starData[:4000]
    starData.to_csv(f'data/raw/new_star_class_data/{starClass}_final_master_data.csv')