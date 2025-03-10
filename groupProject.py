import pandas as pd
import seaborn as seaborn
import matplotlib.pyplot as plt

#Load Data
file_path = '/Users/vusondeptrai/Desktop/TOTAL_KSI_6386614326836635957.csv'
df_Group1 = pd.read_csv(file_path)

#Data Exploration
print(df_Group1.info())
print(df_Group1.dtypes)
print(df_Group1.head())
print(df_Group1.shape)
print(df_Group1.columns)
print(df_Group1.describe())

#Statistics Assessments
print('Mean:', df_Group1.mean(numeric_only=True))
print('Median:', df_Group1.median(numeric_only=True))
print('Mode:', df_Group1.mode().iloc[0])
print('Variance:', df_Group1.var(numeric_only=True))
print('Correlations: ',df_Group1.corr(numeric_only=True))
print('Standard Deviation:', df_Group1.std(numeric_only= True))

#Missing Values
print('Missing values per column: \n', df_Group1.isnull().sum())
plt.figure(figsize=(10,6))
seaborn.heatmap(df_Group1.isnull(), cmap="Blues", cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

#Graphs and Visualization
df_Group1_numeric = df_Group1.select_dtypes(include = ['int64', 'float64'])

#Boxplot
plt.figure(figsize=(12, 6))
df_Group1_numeric.boxplot(rot=45)
plt.title("Boxplot of Numeric Columns")
plt.show()

#Histogram
df_Group1_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Histograms of Numeric Columns")
plt.show()

#Pairplot
subset_cols = ['LATITUDE', 'LONGITUDE', 'TIME', 'FATAL_NO']
seaborn.pairplot(df_Group1[subset_cols], diag_kind="hist")
plt.show()

#Correlation heat map
plt.figure(figsize=(12, 8))
seaborn.heatmap(df_Group1_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


