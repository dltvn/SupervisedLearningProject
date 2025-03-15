import os.path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

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
#plt.show()

# Graphs and Visualization
df_Group1_numeric = df_Group1.select_dtypes(include=['int64', 'float64'])

# Boxplot
plt.figure(figsize=(12, 6))
df_Group1_numeric.boxplot(rot=45)
plt.title("Boxplot of Numeric Columns (Outlier Detection)")
plt.ylabel("Values")
plt.xticks(rotation=45)
#plt.show()

# Histogram
df_Group1_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numeric Features")
#plt.show()

# Pairplot (Limited to Key Features)
subset_cols = ['LATITUDE', 'LONGITUDE', 'TIME', 'FATAL_NO']
sns.pairplot(df_Group1[subset_cols], diag_kind="hist")
plt.suptitle("Pairwise Relationships of Key Features", y=1.02)
#plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_Group1_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Variables")
#plt.show()

# Drop unnecessary columns. !!!Will be updated in part 2 using feature importance
df_Group1.drop(columns=['OBJECTID', 'INDEX', 'ACCNUM'], inplace=True)

# Define target
target_col = 'ACCLASS'

# Drop rows where class is missing (there is 1)
df_Group1 = df_Group1[df_Group1[target_col].notna()]

# split into X and y
X = df_Group1.drop(columns=[target_col])
y = df_Group1[target_col]

# Identify categorical and numerical columns and indices
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_indices = list(range(len(numerical_columns)))
categorical_indices = list(range(len(numerical_columns), len(numerical_columns + categorical_columns)))

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42,stratify=y)

# imputer before resampling as it doesn't work with missing values
imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='mean'), numerical_columns),
    ('cat_imputer', SimpleImputer(strategy='most_frequent'), categorical_columns)
])

# one hot and scaler after resampling as it should be done on raw data
encoder_scaler = ColumnTransformer([
    ('num_scaler', StandardScaler(), numerical_indices),
    ('cat_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_indices)
])

# full pipeline: imputer->resampling->one hot and scaling->classifier
# logistic regression for pipeline testing purposes, best model will be chosen in deliverable 2
training_pipeline = Pipeline([
    ('imputer', imputer),
    ('smote', SMOTENC(categorical_features=categorical_indices, random_state=42)),
    ('encoder_scaler', encoder_scaler),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

training_pipeline.fit(X_train, y_train)
print(training_pipeline.score(X_train, y_train))
# may throw a warning as test data might have categories not present in the training data
print(training_pipeline.score(X_test, y_test))