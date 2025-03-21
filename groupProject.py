import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
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

# Missing values countplot
plt.figure(figsize=(12, 6))
missing_values_plot = missing_values[missing_values > 0].plot(kind='bar')
plt.title('Missing Values Count by Column')
plt.xlabel('Columns')
plt.ylabel('Number of Missing Values')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Graphs and Visualization
df_Group1_numeric = df_Group1.select_dtypes(include=['int64', 'float64'])

# Boxplots
plt.figure(figsize=(12, 6))
df_Group1_numeric.boxplot(rot=45)
plt.title("Boxplot of Numeric Columns (Outlier Detection)")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()
# Individual (for better interpretability)
# for col in df_Group1_numeric.columns:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(y=df_Group1_numeric[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

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

# Class distribution countplot
sns.countplot(x=df_Group1["ACCLASS"])
plt.title('Class Distribution Before SMOTENC')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.show()

# Drop unnecessary columns. x,y are directly correlated with lat and long so dropping to avoid collinearity !!!Will be updated in part 2 using feature importance
df_Group1.drop(columns=['OBJECTID', 'INDEX', 'ACCNUM', 'x', 'y', 'DIVISION'], inplace=True)

# Define target
target_col = 'ACCLASS'

# Drop rows where class is missing (there is 1)
df_Group1 = df_Group1[df_Group1[target_col].notna()]
# Drop rows where class is Property Damage 0 as we are only interested in fatalities
df_Group1 = df_Group1[df_Group1[target_col] != 'Property Damage O']

# split into X and y
X = df_Group1.drop(columns=[target_col])
y = df_Group1[target_col]

# Identify categorical and numerical columns and indices
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_indices = list(range(len(numerical_columns)))
categorical_indices = list(range(len(numerical_columns), len(numerical_columns + categorical_columns)))

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Impute missing values
imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='mean'), numerical_columns),
    ('cat_imputer', SimpleImputer(strategy='most_frequent'), categorical_columns)
])

# Apply imputer to both train and test sets
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Apply SMOTENC
smote = SMOTENC(categorical_features=categorical_indices, random_state=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_imputed, y_train)

# Step 3: Display class distribution after SMOTENC
plt.figure(figsize=(8, 6))
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution After SMOTENC')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.show()

# Step 4: Create and fit the final pipeline (encoder/scaler + classifier)
final_pipeline = Pipeline([
    ('encoder_scaler', ColumnTransformer([
        ('num_scaler', StandardScaler(), numerical_indices),
        ('cat_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_indices)
    ])),
    ('classifier', LogisticRegression(max_iter=1000, random_state=1))
])

# Fit the final pipeline
final_pipeline.fit(X_train_resampled, y_train_resampled)

# Print scores
print("\nTraining Score:", final_pipeline.score(X_train_resampled, y_train_resampled))
print("Test Score:", final_pipeline.score(X_test_imputed, y_test))