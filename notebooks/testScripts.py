import pandas as pd

masterDf = pd.read_csv('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv')
masteSpectralClassCounts = masterDf['StarClass'].value_counts()
print("Master Data File Class Infographic:")
print(masteSpectralClassCounts)

masterDf = pd.read_csv('data/augmentation/finalProcessedSpectrumData/star/starClassMetaDataIndex.csv')
masteSpectralClassCounts = masterDf['StarClass'].value_counts()
print("Master Data File Class Infographic:")
print(masteSpectralClassCounts)