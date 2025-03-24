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
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# to make pandas print dataframes wider
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Load Data
file_path = os.path.join(os.getcwd(), "dataset.csv")
df_Group1 = pd.read_csv(file_path)

# Remove duplicates
df_Group1.drop_duplicates(inplace=True)

# Standardize column names
df_Group1.columns = df_Group1.columns.str.lower().str.replace(' ', '_')

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
print(df_Group1['acclass'].value_counts())

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
plt.close()

# Graphs and Visualization
df_Group1_numeric = df_Group1.select_dtypes(include=['int64', 'float64'])

# Boxplots
plt.figure(figsize=(12, 6))
df_Group1_numeric.boxplot(rot=45)
plt.title("Boxplot of Numeric Columns (Outlier Detection)")
plt.ylabel("Values")
plt.xticks(rotation=45)
plt.show()
plt.close()

# Histogram
df_Group1_numeric.hist(figsize=(12, 10), bins=30, edgecolor='black')
plt.suptitle("Distribution of Numeric Features")
plt.tight_layout()
plt.show()
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_Group1_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numeric Variables")
plt.tight_layout()
plt.show()
plt.close()

# Class distribution countplot
plt.figure(figsize=(10, 6))
sns.countplot(x=df_Group1['acclass'])
plt.title('Class Distribution Before SMOTENC')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()

# Drop unnecessary columns initially
# Remove object_id, index, and accnum as they're just identifiers
df_Group1.drop(columns=['objectid', 'index', 'accnum'], inplace=True)

# Define target
target_col = 'acclass'

# Drop rows where class is missing (there is 1)
df_Group1 = df_Group1[df_Group1[target_col].notna()]
# Drop rows where class is Property Damage 0 as we are only interested in fatalities
df_Group1 = df_Group1[df_Group1[target_col] != 'Property Damage O']

# split into X and y
X = df_Group1.drop(columns=[target_col])
y = df_Group1[target_col]

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Print columns for reference
print("\n===== FEATURE TYPES =====")
print("Numerical columns:", numerical_columns)
print("Categorical columns:", categorical_columns)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# ===== FEATURE SELECTION =====

# Step 1: Handle missing values before feature selection
imputer = ColumnTransformer([
    ('num_imputer', SimpleImputer(strategy='mean'), numerical_columns),
    ('cat_imputer', SimpleImputer(strategy='most_frequent'), categorical_columns)
])

X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=numerical_columns + categorical_columns
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test),
    columns=numerical_columns + categorical_columns
)

print("\n===== FEATURE SELECTION USING 3 METHODS =====")

# METHOD 1: Mutual Information (for both numeric and categorical features)
print("\n=== METHOD 1: MUTUAL INFORMATION ===")

# First encode categorical features for mutual information
encoder_mi = ColumnTransformer(
    [('cat_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough'
)
X_train_encoded_mi = encoder_mi.fit_transform(X_train_imputed)
feature_names_mi = encoder_mi.get_feature_names_out()

# Apply mutual information
selector_mi = SelectKBest(mutual_info_classif, k='all')
selector_mi.fit(X_train_encoded_mi, y_train)

# Get mutual information scores
mi_scores = pd.DataFrame({
    'Feature': feature_names_mi,
    'MI_Score': selector_mi.scores_
})
mi_scores = mi_scores.sort_values('MI_Score', ascending=False)
print("\n** Mutual Information Feature Scores **")
print(mi_scores.head(15).to_string())

# Visualize mutual information scores (top 10)
plt.figure(figsize=(12, 8))
top_mi_features = mi_scores.head(10)
sns.barplot(x='MI_Score', y='Feature', data=top_mi_features)
plt.title('Top 10 Features by Mutual Information')
plt.tight_layout()
plt.show()
plt.close()

# METHOD 2: Random Forest Feature Importance (as requested)
print("\n=== METHOD 2: RANDOM FOREST FEATURE IMPORTANCE ===")

# First, one-hot encode categorical features for the Random Forest
encoder_rf = ColumnTransformer(
    [('cat_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_columns)],
    remainder='passthrough'
)

X_train_encoded_rf = encoder_rf.fit_transform(X_train_imputed)
feature_names_rf = encoder_rf.get_feature_names_out()

# Train a Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=1)
rf.fit(X_train_encoded_rf, y_train)

# Get feature importances
feature_importance = pd.DataFrame({
    'Feature': feature_names_rf,
    'Importance': rf.feature_importances_
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\n** Random Forest Feature Importance **")
print(feature_importance.head(15).to_string())  # Show top 15 features

# Visualize Random Forest feature importance (top 10)
plt.figure(figsize=(12, 8))
top_rf_features = feature_importance.head(10)
sns.barplot(x='Importance', y='Feature', data=top_rf_features)
plt.title('Top 10 Features by Random Forest Importance')
plt.tight_layout()
plt.show()
plt.close()

# METHOD 3: Correlation Analysis with Target (for numerical features)
print("\n=== METHOD 3: CORRELATION WITH TARGET ===")

# Create a dataframe with target for correlation analysis
y_numeric = pd.get_dummies(y, drop_first=True)  # Convert categorical target to numeric
target_columns = y_numeric.columns

# Concatenate features and target for correlation analysis
X_train_with_target = pd.concat([X_train_imputed[numerical_columns], y_numeric], axis=1)

# Calculate correlation matrix
correlation_with_target = {}
for target_col in target_columns:
    # Calculate correlation of each numerical feature with this target class
    corr_values = X_train_with_target[numerical_columns].corrwith(X_train_with_target[target_col]).abs()

    # Store in dictionary
    for feature, corr in corr_values.items():
        if feature in correlation_with_target:
            correlation_with_target[feature] = max(correlation_with_target[feature], corr)
        else:
            correlation_with_target[feature] = corr

# Convert to DataFrame
corr_scores = pd.DataFrame({
    'Feature': list(correlation_with_target.keys()),
    'Correlation': list(correlation_with_target.values())
})
corr_scores = corr_scores.sort_values('Correlation', ascending=False)
print("\n** Feature Correlation with Target **")
print(corr_scores.to_string())

# Visualize correlation scores
plt.figure(figsize=(12, 8))
sns.barplot(x='Correlation', y='Feature', data=corr_scores)
plt.title('Numerical Features by Correlation with Target')
plt.tight_layout()
plt.show()
plt.close()

print("\n===== FEATURE SELECTION CONSENSUS =====")

# Get top features from each method
top_mi_features = mi_scores.head(5)['Feature'].tolist()
top_rf_features = feature_importance.head(5)['Feature'].tolist()
top_corr_features = corr_scores.head(5)['Feature'].tolist()


# Check for column name differences due to encoding
# Extract original column names from encoded features
def extract_original_feature(encoded_feature):
    # For features from OneHotEncoder, extract the original column name
    if 'cat_encoder__' in encoded_feature:
        # Format: cat_encoder__column_name_value
        parts = encoded_feature.split('__')
        if len(parts) > 1:
            column_and_value = parts[1]
            # Extract column name (before the last underscore)
            return '_'.join(column_and_value.split('_')[:-1])
    # For numeric features (passthrough in encoder)
    elif 'remainder__' in encoded_feature:
        return encoded_feature.replace('remainder__', '')
    # Return as is if can't parse
    return encoded_feature


# Extract original feature names
original_mi_features = [extract_original_feature(feat) for feat in top_mi_features]
original_rf_features = [extract_original_feature(feat) for feat in top_rf_features]

# For numerical features from correlation, they're already in original form
original_corr_features = top_corr_features

# Count appearances across methods
feature_counts = {}
for feature_list in [original_mi_features, original_rf_features, original_corr_features]:
    for feature in feature_list:
        if feature in feature_counts:
            feature_counts[feature] += 1
        else:
            feature_counts[feature] = 1

# Convert to DataFrame and sort
feature_consensus = pd.DataFrame({
    'Feature': list(feature_counts.keys()),
    'Methods_Selected': list(feature_counts.values())
})
feature_consensus = feature_consensus.sort_values(['Methods_Selected', 'Feature'], ascending=[False, True])

print("\n** Feature Selection Consensus **")
print(feature_consensus.to_string())

# Check for collinearity in numerical features
print("\nChecking for Collinearity Among Numerical Features:")
corr_matrix = X_train_imputed[numerical_columns].corr().abs()
high_corr_pairs = []

for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.7:  # Threshold for high correlation
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

if high_corr_pairs:
    print("\nHighly Correlated Feature Pairs (correlation > 0.7):")
    for feat1, feat2, corr in high_corr_pairs:
        print(f"{feat1} and {feat2}: {corr:.3f}")
else:
    print("No high correlation pairs found among numerical features.")

print("\n===== FEATURE SELECTION RECOMMENDATION =====")

# Get features selected by at least 2 methods or with very high importance in one method
selected_by_multiple = feature_consensus[feature_consensus['Methods_Selected'] >= 2]['Feature'].tolist()
highly_important_in_rf = [extract_original_feature(feat) for feat in feature_importance.head(5)['Feature'].tolist()]
highly_important_in_mi = [extract_original_feature(feat) for feat in mi_scores.head(5)['Feature'].tolist()]

# Combine and remove duplicates
recommended_features = list(set(selected_by_multiple + highly_important_in_rf + highly_important_in_mi))

# For categorical features, we need to keep entire column
categorical_to_keep = [col for col in categorical_columns if any(col in feat for feat in recommended_features)]
numerical_to_keep = [col for col in numerical_columns if col in recommended_features]

# Final list of recommended features
final_recommended = list(set(numerical_to_keep + categorical_to_keep))

print("\nRECOMMENDED FEATURES TO KEEP:")
for feature in final_recommended:
    methods = []
    if feature in numerical_columns:
        if feature in original_mi_features: methods.append("Mutual Information")
        if feature in original_rf_features: methods.append("Random Forest")
        if feature in original_corr_features: methods.append("Correlation Analysis")
    else:  # Categorical feature
        # Check if any encoding of this feature is important
        if any(feature in feat for feat in original_mi_features): methods.append("Mutual Information")
        if any(feature in feat for feat in original_rf_features): methods.append("Random Forest")

    print(f"- {feature} (Selected by: {', '.join(methods)})")

# Features to remove
all_features = numerical_columns + categorical_columns
features_to_remove = [feat for feat in all_features if feat not in final_recommended]

print("\nFEATURES TO CONSIDER REMOVING:")
for feature in features_to_remove:
    print(f"- {feature}")

# Remove highly correlated features from recommended list
if high_corr_pairs:
    print("\nFrom correlated feature pairs, consider removing one from each pair:")
    for feat1, feat2, corr in high_corr_pairs:
        # Check if both are in recommended list
        if feat1 in final_recommended and feat2 in final_recommended:
            # Compare feature importance to decide which to keep
            feat1_importance = 0
            feat2_importance = 0

            # Check importance in RF
            for idx, row in feature_importance.iterrows():
                if feat1 in row['Feature']:
                    feat1_importance += row['Importance']
                if feat2 in row['Feature']:
                    feat2_importance += row['Importance']

            if feat1_importance > feat2_importance:
                print(f"- Consider removing {feat2} (correlated with {feat1}, but less important)")
            else:
                print(f"- Consider removing {feat1} (correlated with {feat2}, but less important)")

print("\n===== FINAL MODEL WITH SELECTED FEATURES =====")

# Filter to keep only recommended features
X_train_selected = X_train_imputed[final_recommended]
X_test_selected = X_test_imputed[final_recommended]

# Identify categorical columns in the selected dataset
categorical_selected = [col for col in X_train_selected.columns if col in categorical_columns]
numerical_selected = [col for col in X_train_selected.columns if col in numerical_columns]

# Categorical indices for SMOTE
categorical_indices = [i for i, col in enumerate(X_train_selected.columns) if col in categorical_selected]

# Apply SMOTENC with selected features
smote = SMOTENC(categorical_features=categorical_indices, random_state=1)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_selected, y_train)

# Create a final pipeline with selected features
final_pipeline = Pipeline([
    ('encoder_scaler', ColumnTransformer([
        ('num_scaler', StandardScaler(), [i for i, col in enumerate(X_train_selected.columns)
                                          if col in numerical_selected]),
        ('cat_encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'),
         [i for i, col in enumerate(X_train_selected.columns) if col in categorical_selected])
    ]) if categorical_selected else ('num_scaler', StandardScaler())),
    ('classifier', LogisticRegression(max_iter=1000, random_state=1))
])

# Fit the final pipeline
final_pipeline.fit(X_train_resampled, y_train_resampled)

# Print scores
print("\nTraining Score:", final_pipeline.score(X_train_resampled, y_train_resampled))
print("Test Score:", final_pipeline.score(X_test_selected, y_test))

# Visualize class distribution after SMOTENC
plt.figure(figsize=(10, 6))
sns.countplot(x=y_train_resampled)
plt.title('Class Distribution After SMOTENC')
plt.xlabel('ACCLASS')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.close()

# Feature importance visualization for final visualization
plt.figure(figsize=(12, 8))
plt.title('Selected Features Overview')
plt.barh([f"{feat} (Selected by {feature_counts.get(feat, 0)} methods)" for feat in final_recommended],
         [feature_counts.get(feat, 0) for feat in final_recommended])
plt.xlabel('Number of Selection Methods')
plt.ylabel('Selected Features')
plt.tight_layout()
plt.show()
plt.close()


# Conclusion analysis in the code so other group member who do the documentation part can base on it
# and have other perspective analysis
print("\n===== CONCLUSION =====")
print(
    f"Feature selection has been applied using 3 methods: Mutual Information, Random Forest, and Correlation Analysis.")
print(f"From the original {len(all_features)} features, {len(final_recommended)} important features were selected.")
print("These selected features have been used to build the final model.")
print(f"The model accuracy on the test set is: {final_pipeline.score(X_test_selected, y_test):.4f}")