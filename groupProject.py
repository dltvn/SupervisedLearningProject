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
from scipy.stats import uniform, randint

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


#! RANDOM FOREST FEATURE IMPORTANCE
# Temporarily ordinal encode X for feature selection
feature_selection_transformer = ColumnTransformer([
    ('cat_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns)
], remainder='passthrough')
X_fs = feature_selection_transformer.fit_transform(X_imputed_df)
# Convert to DataFrame for RandomForestClassifier
X_fs_df = pd.DataFrame(X_fs, columns=[name.split('__')[-1] for name in feature_selection_transformer.get_feature_names_out()], index=X.index)
# Fit data to random forest for feature selection
rf_clf = RandomForestClassifier(n_estimators=100, random_state=1)
rf_clf.fit(X_fs_df, y)
# Feature importances in descending order
feature_importances = pd.Series(rf_clf.feature_importances_, index=X_fs_df.columns)
feature_importances_sorted = feature_importances.sort_values(ascending=False)
# Select top N features
N = 15
top_features = feature_importances_sorted.head(N).index.tolist()
print(f"\nSelected Top {N} Features:\n", top_features)

# Plotting top features by importance score
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances_sorted.values, y=feature_importances_sorted.index)
plt.title('Features by Importance')
plt.xlabel('Importance Score')
plt.show()
# Extract selected features from imputed(not encoded) X
X_selected = X_imputed_df[top_features]

#! NUM AND CAT UPDATE
# Update numerical and categorical columns
numerical_selected = [col for col in top_features if col in numerical_columns]
categorical_selected = [col for col in top_features if col in categorical_columns]

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
# Scaler and one hot for training
one_hot = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore');
one_hot_transformer = ColumnTransformer([
    ('cat_encoder', one_hot , categorical_selected)
], remainder="passthrough")
scaler_one_hot = ColumnTransformer([
    ('num_scaler', StandardScaler(), numerical_selected),
    ('cat_encoder', one_hot , categorical_selected)
])

#! MODEL PIPELINES
# Model pipeline dictionary
pipelines = {
    'Logistic Regression': Pipeline([
        ('preprocessor', scaler_one_hot),
        ('classifier', LogisticRegression(max_iter=1000, random_state=1))
    ]),
    'Decision Tree': Pipeline([
        ('preprocessor', one_hot_transformer),
        ('classifier', DecisionTreeClassifier(random_state=1))
    ]),
    'SVM': Pipeline([
        ('preprocessor', scaler_one_hot),
        ('classifier', SVC(probability=True, random_state=1))
    ]),
    'Random Forest': Pipeline([
        ('preprocessor', one_hot_transformer),
        ('classifier', RandomForestClassifier(random_state=1))
    ]),
    'Neural Network': Pipeline([
        ('preprocessor', scaler_one_hot),
        ('classifier', MLPClassifier(max_iter=500, random_state=1))
    ])
}
best_models = {}

#! GRID SEARCH
# # Logistic Regression Grid Search
# log_reg_params = {
#     'classifier__C': [0.01, 0.1, 1, 10, 100],
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__solver': ['liblinear']
# }
# log_reg_grid = GridSearchCV(pipelines['Logistic Regression'], log_reg_params, cv=5, scoring='accuracy')
# log_reg_grid.fit(X_train_resampled, y_train_resampled)

# # Decision Tree Grid Search
# dt_params = {
#     'classifier__criterion': ['gini', 'entropy'],
#     'classifier__max_depth': [None, 5, 10, 20],
#     'classifier__min_samples_split': [2, 5, 10]
# }
# dt_grid = GridSearchCV(pipelines['Decision Tree'], dt_params, cv=5, scoring='accuracy')
# dt_grid.fit(X_train_resampled, y_train_resampled)

# # Random Forest Grid Search
# rf_params = {
#     'classifier__n_estimators': [50, 100, 200],
#     'classifier__criterion': ['gini', 'entropy'],
#     'classifier__max_depth': [None, 10, 20],
#     'classifier__min_samples_split': [2, 5, 10]
# }
# rf_grid = GridSearchCV(pipelines['Random Forest'], rf_params, cv=5, scoring='accuracy')
# rf_grid.fit(X_train_resampled, y_train_resampled)

# # SVM Grid Search
# svm_params = {
#     'classifier__C': [0.1, 1, 10],
#     'classifier__kernel': ['linear', 'rbf'],
#     'classifier__gamma': ['scale', 'auto']
# }
# svm_grid = GridSearchCV(pipelines['SVM'], svm_params, cv=5, scoring='accuracy')
# svm_grid.fit(X_train_resampled, y_train_resampled)

# # Neural Network (MLP) Grid Search
# mlp_params = {
#     'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50)],
#     'classifier__activation': ['relu', 'tanh'],
#     'classifier__solver': ['adam'],
#     'classifier__alpha': [0.0001, 0.001],
#     'classifier__learning_rate': ['constant', 'adaptive']
# }
# mlp_grid = GridSearchCV(pipelines['Neural Network'], mlp_params, cv=5, scoring='accuracy')
# mlp_grid.fit(X_train_resampled, y_train_resampled)

# # Print best parameters and add best model to the dictionary
# print("Best Logistic Regression:", log_reg_grid.best_params_)
# best_models["Logistic Regression"] = log_reg_grid.best_estimator_
# print("Best Decision Tree:", dt_grid.best_params_)
# best_models["Decision Tree"] = dt_grid.best_estimator_
# print("Best Random Forest:", rf_grid.best_params_)
# best_models["Random Forest"] = rf_grid.best_estimator_
# print("Best SVM:", svm_grid.best_params_)
# best_models["SVM"] = svm_grid.best_estimator_
# print("Best MLP:", mlp_grid.best_params_)
# best_models["MLP"] = mlp_grid.best_estimator_

# #! RANDOMIZED SEARCH
# # Logistic Regression Randomized Search
# log_reg_dist = {
#     'classifier__C': uniform(0.01, 100),
#     'classifier__penalty': ['l1', 'l2'],
#     'classifier__solver': ['liblinear']
# }
# log_reg_rand = RandomizedSearchCV(pipelines['Logistic Regression'], log_reg_dist, n_iter=10, cv=5, scoring='accuracy', random_state=1)
# log_reg_rand.fit(X_train_resampled, y_train_resampled)

# # Decision Tree Randomized Search
# dt_dist = {
#     'classifier__criterion': ['gini', 'entropy'],
#     'classifier__max_depth': [None] + list(range(5, 30)),
#     'classifier__min_samples_split': randint(2, 20)
# }
# dt_rand = RandomizedSearchCV(pipelines['Decision Tree'], dt_dist, n_iter=10, cv=5, scoring='accuracy', random_state=1)
# dt_rand.fit(X_train_resampled, y_train_resampled)

# # Random Forest Randomized Search
# rf_dist = {
#     'classifier__n_estimators': randint(50, 500),
#     'classifier__criterion': ['gini', 'entropy'],
#     'classifier__max_depth': [None] + list(range(10, 50)),
#     'classifier__min_samples_split': randint(2, 20)
# }
# rf_rand = RandomizedSearchCV(pipelines['Random Forest'], rf_dist, n_iter=10, cv=5, scoring='accuracy', random_state=1)
# rf_rand.fit(X_train_resampled, y_train_resampled)

# # SVM Randomized Search
# svm_dist = {
#     'classifier__C': uniform(0.1, 10),
#     'classifier__kernel': ['linear', 'rbf'],
#     'classifier__gamma': ['scale', 'auto']
# }
# svm_rand = RandomizedSearchCV(pipelines['SVM'], svm_dist, n_iter=10, cv=5, scoring='accuracy', random_state=1)
# svm_rand.fit(X_train_resampled, y_train_resampled)

# # Neural Network (MLP) Randomized Search
# mlp_dist = {
#     'classifier__hidden_layer_sizes': [(50,), (100,), (150, 75, 50)],
#     'classifier__activation': ['relu', 'tanh'],
#     'classifier__solver': ['adam'],
#     'classifier__alpha': uniform(0.0001, 0.01),
#     'classifier__learning_rate': ['constant', 'adaptive']
# }
# mlp_rand = RandomizedSearchCV(pipelines['Neural Network'], mlp_dist, n_iter=10, cv=5, scoring='accuracy', random_state=1)
# mlp_rand.fit(X_train_resampled, y_train_resampled)

# # Print best parameters and add best model to the dictionary
# print("Best Logistic Regression:", log_reg_rand.best_params_)
# best_models["Logistic Regression"] = log_reg_rand.best_estimator_
# print("Best Decision Tree:", dt_rand.best_params_)
# best_models["Decision Tree"] = dt_rand.best_estimator_
# print("Best Random Forest:", rf_rand.best_params_)
# best_models["Random Forest"] = rf_rand.best_estimator_
# print("Best SVM:", svm_rand.best_params_)
# best_models["SVM"] = svm_rand.best_estimator_
# print("Best MLP:", mlp_rand.best_params_)
# best_models["MLP"] = mlp_rand.best_estimator_

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

# #! SCORING BEST MODELS
# results = {}
# for model_name, model in best_models.items():
#     print(f"\nTraining {model_name}...")
#     results[model_name] = evaluate_model(model, X_test, y_test, model_name)

#! SCORING A TEST MODEL
log_reg_test = pipelines['Logistic Regression']
log_reg_test.fit(X_train_resampled, y_train_resampled)
print(evaluate_model(log_reg_test, X_test, y_test, "Logistic Regression"))
