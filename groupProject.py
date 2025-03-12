import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# to make pandas print dataframes wider
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load Data
file_path = os.path.join(os.getcwd(), "dataset.csv")
df_Group1 = pd.read_csv(file_path)

# Data Exploration
print("\n===== DATA EXPLORATION =====")
print("\n** Dataset Information **")
print(df_Group1.info())

print("\n** Data Types **")
print(df_Group1.dtypes)

print("\n** First 5 Rows of Data **")
print(df_Group1.head().to_string())

print("\n** Dataset Shape (Rows, Columns) **")
print(df_Group1.shape)

print("\n** Column Names **")
print(df_Group1.columns.tolist())

print("\n** Class Counts **")
print(df_Group1["ACCLASS"].value_counts())

print("\n** Summary Statistics **")
print(df_Group1.describe().transpose().to_string())

# Statistics Assessments
print("\n===== STATISTICAL ASSESSMENTS =====")
print("\n** Mean Values **")
print(df_Group1.mean(numeric_only=True).round(2).to_string())

print("\n** Median Values **")
print(df_Group1.median(numeric_only=True).to_string())

print("\n** Mode (First Occurrence) **")
print(df_Group1.mode().iloc[0].to_string())

print("\n** Variance **")
print(df_Group1.var(numeric_only=True).round(2).to_string())

print("\n** Correlation Matrix **")
print(df_Group1.corr(numeric_only=True).round(2).to_string())

print("\n** Standard Deviation **")
print(df_Group1.std(numeric_only=True).round(2).to_string())

# Missing Values
print("\n===== MISSING VALUES ASSESSMENT =====")
missing_values = df_Group1.isnull().sum()
print("\n** Missing Values Per Column **")
print(missing_values[missing_values > 0].to_string())

plt.figure(figsize=(10, 6))
sns.heatmap(df_Group1.isnull(), cmap="Blues", cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

# Graphs and Visualization
df_Group1_numeric = df_Group1.select_dtypes(include=['int64', 'float64'])

# Boxplot
plt.figure(figsize=(12, 6))
df_Group1_numeric.boxplot(rot=45)
plt.title("Boxplot of Numeric Columns (Outlier Detection)")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()

# Histogram
df_Group1_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numeric Features")
plt.show()

# Pairplot (Limited to Key Features)
subset_cols = ['LATITUDE', 'LONGITUDE', 'TIME', 'FATAL_NO']
sns.pairplot(df_Group1[subset_cols], diag_kind="hist")
plt.suptitle("Pairwise Relationships of Key Features", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_Group1_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Variables")
plt.show()
