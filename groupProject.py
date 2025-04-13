import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import RocCurveDisplay, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pickle
from scipy.stats import uniform, randint
from boruta import BorutaPy
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.base import BaseEstimator, TransformerMixin
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore", message="Found unknown categories.*")
np.random.seed(1)
import random
random.seed(1)

# to make pandas print dataframes wider
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load Data
file_path = os.path.join(os.getcwd(), "dataset.csv")
df_Group1 = pd.read_csv(file_path)

df_Group1_numeric = df_Group1.select_dtypes(include=['int64', 'float64'])

#! EXPLORATION
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

print("\n** Summary Statistics **")
print(df_Group1.describe().transpose().to_string())

#! STATISTICS + STATISTICAL PLOTS
# Statistics Assessments
print("\n** Mean Values **")
print(df_Group1.mean(numeric_only=True).round(2).to_string())
# Mean Values of Numeric Features Grouped by ACCLASS
grouped_means = df_Group1.groupby('ACCLASS')[df_Group1_numeric.columns.tolist()].mean()
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_means.transpose(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Mean Values of Numeric Features Grouped by ACCLASS")
plt.xlabel("ACCLASS")
plt.ylabel("Features")
plt.show()

print("\n** Median Values **")
print(df_Group1.median(numeric_only=True).to_string())
# Median Values of Numeric Features Grouped by ACCLASS
grouped_medians = df_Group1.groupby('ACCLASS')[df_Group1_numeric.columns.tolist()].median()
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_medians.transpose(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Median Values of Numeric Features Grouped by ACCLASS")
plt.xlabel("ACCLASS")
plt.ylabel("Features")
plt.show()

print("\n** Mode (First Occurrence) **")
print(df_Group1.mode().iloc[0].to_string())



print("\n** Variance **")
print(df_Group1.var(numeric_only=True).round(2).to_string())
# Variance of Numeric Features Grouped by ACCLASS
grouped_variances = df_Group1.groupby('ACCLASS')[df_Group1_numeric.columns.tolist()].var()
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_variances.transpose(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Variance of Numeric Features Grouped by ACCLASS")
plt.xlabel("ACCLASS")
plt.ylabel("Features")
plt.show()

print("\n** Standard Deviation **")
print(df_Group1.std(numeric_only=True).round(2).to_string())
# Standard Deviation of Numeric Features Grouped by ACCLASS
grouped_stds = df_Group1.groupby('ACCLASS')[df_Group1_numeric.columns.tolist()].std()
plt.figure(figsize=(12, 6))
sns.heatmap(grouped_stds.transpose(), annot=True, fmt=".2f", cmap="YlGnBu")
plt.title("Standard Deviation of Numeric Features Grouped by ACCLASS")
plt.xlabel("ACCLASS")
plt.ylabel("Features")
plt.show()

#! MISSING VALUES
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

#! CLASS BALANCE/DISTRIBUTION
# Class distribution
print("\n** Class Counts **")
print(df_Group1["ACCLASS"].value_counts())
# Class distribution countplot
sns.countplot(x=df_Group1["ACCLASS"])
plt.title('Class Distribution Before SMOTENC')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.show()

#! OTHER PLOTS
# Histogram
df_Group1_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numeric Features")
plt.show()

#? Pairplot (Limited to Key Features)
subset_cols = ['LATITUDE', 'LONGITUDE', 'TIME', 'FATAL_NO']
sns.pairplot(df_Group1[subset_cols], diag_kind="hist")
plt.suptitle("Pairwise Relationships of Key Features", y=1.02)
plt.show()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_Group1_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Variables")
plt.show()

# Map
plt.figure(figsize=(10, 8))
sns.scatterplot(x='LONGITUDE', y='LATITUDE', hue='ACCLASS', data=df_Group1, alpha=0.6)
plt.title('Geospatial Distribution of Accidents by ACCLASS')
plt.show()
# Density Plot using kde
plt.figure(figsize=(10, 8))
sns.kdeplot(data=df_Group1,x='LONGITUDE',y='LATITUDE',hue='ACCLASS',alpha=0.5,fill=True)
plt.title('Geospatial Density of Accidents by ACCLASS')
plt.show()

# Trend of ACCLASS over Years
df_Group1['YEAR'] = pd.to_datetime(df_Group1['DATE']).dt.year
df_Group1.groupby('YEAR')['ACCLASS'].value_counts().unstack().plot(kind='line')
plt.title('Trend of ACCLASS over Years')
plt.ylabel('Count')
plt.show()

# Trend of ACCLASS over Months
df_Group1['MONTH'] = pd.to_datetime(df_Group1['DATE']).dt.month
df_Group1.groupby('MONTH')['ACCLASS'].value_counts().unstack().plot(kind='line')
plt.title('Trend of ACCLASS over Months')
plt.ylabel('Count')
plt.show()

# Trend of ACCLASS over Hours
df_Group1['HOUR'] = pd.to_datetime(df_Group1['DATE']).dt.hour
df_Group1.groupby('HOUR')['ACCLASS'].value_counts().unstack().plot(kind='line')
plt.title('Trend of ACCLASS over Hours')
plt.ylabel('Count')
plt.show()


#! DROP ROWS
# Drop rows where class is missing (there is 1)
df_Group1 = df_Group1[df_Group1["ACCLASS"].notna()]
# Drop rows where class is Property Damage 0 as we are only interested in fatalities
df_Group1 = df_Group1[df_Group1["ACCLASS"] != 'Property Damage O']

#! DROP COLUMNS
# Drop index columns and x and y as they are perfectly correlated with lat and long
df_Group1.drop(columns=["OBJECTID", "INDEX", "ACCNUM", "x", "y"], inplace=True)

# Plit into X an y
X = df_Group1.drop(columns=["ACCLASS"])
y = df_Group1["ACCLASS"]

# Extract cat and num columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()


#! FILL MISSING VALUES
# Fill missing values
imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='mean'), numerical_columns),
    ('cat_imputer', SimpleImputer(strategy='most_frequent'), categorical_columns)
])
X_imputed = imputer.fit_transform(X)
# Convert back to DataFrame for readability and column names
X_imputed_df = pd.DataFrame(X_imputed, columns=[name.split('__')[-1] for name in imputer.get_feature_names_out()], index=X.index)


#! FEATURE SELECTION
feature_selection_transformer = ColumnTransformer([
    ('cat_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns)
], remainder='passthrough')
X_fs = feature_selection_transformer.fit_transform(X_imputed_df)
# Convert to DataFrame
X_fs_df = pd.DataFrame(X_fs, columns=[name.split('__')[-1] for name in feature_selection_transformer.get_feature_names_out()], index=X.index)

print("\n=== METHOD: Random Forest + Boruta ===")
# Prepare data for Boruta (must be numpy arrays)
X_boruta = X_fs_df.values  # Ordinal encoded + imputed
y_boruta = y.values
# Initialize Boruta with a random forest
rf_for_boruta = RandomForestClassifier(n_estimators=100, random_state=1, n_jobs=-1)
boruta_selector = BorutaPy(estimator=rf_for_boruta, n_estimators='auto', verbose=0, random_state=1)
boruta_selector.fit(X_boruta, y_boruta)
# Get selected features
boruta_selected_features = X_fs_df.columns[boruta_selector.support_].tolist()
print("Random Forest + Boruta Selected Features:")
print(boruta_selected_features)

print("\n=== METHOD: MUTUAL INFORMATION ===")
# Apply Mutual Information
selector_mi = SelectKBest(mutual_info_classif, k='all')
selector_mi.fit(X_fs_df, y)
mi_scores = pd.Series(selector_mi.scores_, index=X_fs_df.columns).sort_values(ascending=False)
mi_selected_features = mi_scores.head(15).index.to_list()
# Plot MI scores
plt.figure(figsize=(12, 8))
sns.barplot(x=mi_scores.values, y=mi_scores.index)
plt.title("Top Features by Mutual Information")
plt.tight_layout()
plt.show()
plt.close()
print("Mutual Information top 15 Features:")
print(mi_selected_features)

print("\n=== METHOD: BORUTA RF + MUTUAL INFORMATION VOTING ===")
voted_features = list(set(boruta_selected_features).intersection(set(mi_selected_features)))
print("Features voted for by both boruta rf and mutual information top 15:")
print(voted_features)

#! UPDATE X
X_selected = X_imputed_df[voted_features]

#! TRANSFORMER FOR ONLY USING SELECTED FEATURES
class FeatureSubsetSelector(BaseEstimator, TransformerMixin):
    def __init__(self, selected_features):
        self.selected_features = selected_features

    def fit(self, X, y=None):
        return self  # no fitting needed

    def transform(self, X):
        return X[self.selected_features]

#! NUM AND CAT UPDATE
# Update numerical and categorical columns
numerical_selected = [col for col in voted_features if col in numerical_columns]
categorical_selected = [col for col in voted_features if col in categorical_columns]


#! TRAIN TEST SPLIT
# Split selected data into train/test 70/30
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=1, stratify=y)

#! BALANCING
# Balance the dataset using SMOTENC (works well with categorical)
smote = SMOTENC(categorical_features=categorical_selected, random_state=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
X_train_resampled = pd.DataFrame(X_train_resampled,columns=smote.get_feature_names_out(),index=None)
# Class distribution countplot
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution After SMOTENC')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.show()

#! SCALER AND ENCODING

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ]), numerical_selected),
    
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ]), categorical_selected)
])

#! MODEL PIPELINES
# Model pipeline dictionary
pipelines = {
    'Logistic Regression': Pipeline([
        ('feature_selector', FeatureSubsetSelector(voted_features)),
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=1))
    ]),
    'Decision Tree': Pipeline([
        ('feature_selector', FeatureSubsetSelector(voted_features)),
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=1))
    ]),
    # Taking too long to train, even with linear kernel
    # 'SVM': Pipeline([
    #     ('feature_selector', FeatureSubsetSelector(voted_features)),
    #     ('preprocessor', preprocessor),
    #     ('classifier', SVC(probability=True, kernel='linear', random_state=1))
    # ]),
    'Random Forest': Pipeline([
        ('feature_selector', FeatureSubsetSelector(voted_features)),
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=1))
    ]),
    'Neural Network': Pipeline([
        ('feature_selector', FeatureSubsetSelector(voted_features)),
        ('preprocessor', preprocessor),
        ('classifier', MLPClassifier(max_iter=500, random_state=1))
    ]),
    'LightGBM': Pipeline([
        ('feature_selector', FeatureSubsetSelector(voted_features)),
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(random_state=1))
    ])
}


#! FUNCTION FOR SCORING
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='Fatal')
    rec = recall_score(y_test, y_pred, pos_label='Fatal')
    f1 = f1_score(y_test, y_pred, pos_label='Fatal')
    auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

    print(f"\n===== {model_name} Performance =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    if auc: print(f"ROC AUC  : {auc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    if y_prob is not None:
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"{model_name} ROC Curve")
        plt.show()

    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 'auc': auc}


#! EVALUATE ALL DEFAULT MODELS
# Evaluate all models in the pipelines dictionary
results = {}

for model_name, model in pipelines.items():
    print(f"\n===== Training and Evaluating: {model_name} =====")
    model.fit(X_train_resampled, y_train_resampled)
    results[model_name] = evaluate_model(model, X_test, y_test, model_name)
    
#! DEFAULT MODEL COMPARISON
print("\n===== METRICS COMPARISON =====")
# Compare models using nested loops directly
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    best_model = None
    best_value = -np.inf

    for model_name in results:
        value = results[model_name][metric]
        if value is not None and value > best_value:
            best_value = value
            best_model = model_name

    print(f"{metric.capitalize()} -> Best: {best_model} ({best_value:.4f})")

# #! DUMPING LOGISTIC REGRESSION MODEL INTO PKL
# with open('best_model.pkl', 'wb') as file:
#     pickle.dump(best_model, file)
