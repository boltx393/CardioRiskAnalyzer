# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import os

# %%
# Create a directory to save visualizations
os.makedirs('visualizations', exist_ok=True)

# Load the data
df = pd.read_csv(r'C:\Users\iamaa\Downloads\dataset_heartdisease\2022\heart_2022_with_nans.csv')

# %%
# 1. Data Exploration
print("Data Features:")
print(df.info())

# %%
print("\nSample data:")
print(df.head())

# %%
print("\nDataset shape:", df.shape)

# %%
print("\nSummary statistics:")
print(df.describe())

# %%
print("\nMissing values:")
missing_values = df.isnull().sum()
print(missing_values)

# %%
# Visualize missing data
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Missing Data in Heart Disease Dataset')
plt.tight_layout()
plt.savefig('visualizations/missing_data_heatmap.png')
plt.show()
plt.close()

# %%
# Percentage of missing values
missing_percentage = (missing_values / len(df)) * 100
print("\nPercentage of missing values:")
print(missing_percentage)

# %%
# Visualize distribution of numeric features
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
fig, axes = plt.subplots(nrows=(len(numeric_features)+1)//3, ncols=3, figsize=(20, 5*((len(numeric_features)+1)//3)))
for i, feature in enumerate(numeric_features):
    sns.histplot(df[feature].dropna(), kde=True, ax=axes[i//3, i%3])
    axes[i//3, i%3].set_title(f'Distribution of {feature}')
plt.tight_layout()
plt.savefig('visualizations/numeric_features_distribution.png')
plt.show()
plt.close()

# %%
# Correlation matrix of numerical features
correlation_matrix = df[numeric_features].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.tight_layout()
plt.savefig('visualizations/correlation_matrix.png')
plt.show()
plt.close()


# %%
# 2. Data Preprocessing
# Separate features and target variable
X = df.drop('HadHeartAttack', axis=1)
y = df['HadHeartAttack']

categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

# Try to create OneHotEncoder with sparse=False, if it fails, create without this parameter
try:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])
except TypeError:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names
try:
    feature_names = (numeric_features.tolist() + 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())
except AttributeError:
    # For older versions of scikit-learn
    feature_names = (numeric_features.tolist() + 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names(categorical_features).tolist())

# Convert to dense array if sparse
if hasattr(X_train_preprocessed, 'toarray'):
    X_train_preprocessed = X_train_preprocessed.toarray()
if hasattr(X_test_preprocessed, 'toarray'):
    X_test_preprocessed = X_test_preprocessed.toarray()

# Create DataFrames with preprocessed data
df_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=feature_names)
df_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names)

print("\nPreprocessed training data shape:", df_train_preprocessed.shape)
print("Preprocessed test data shape:", df_test_preprocessed.shape)

# %%
# 3. Dimensionality Reduction with PCA
pca = PCA()
X_train_pca = pca.fit_transform(df_train_preprocessed)
X_test_pca = pca.transform(df_test_preprocessed)

# Plot explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs. Number of Components')
plt.savefig('visualizations/pca_explained_variance.png')
plt.show()
plt.close()

# %%
# Determine number of components for 95% variance
n_components_95 = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.95) + 1
print(f"\nNumber of components needed for 95% variance: {n_components_95}")

# Apply PCA with 95% variance
pca_95 = PCA(n_components=n_components_95)
X_pca_95 = pca_95.fit_transform(df_train_preprocessed)

# Create a DataFrame with PCA results
df_pca = pd.DataFrame(X_pca_95, columns=[f'PC{i+1}' for i in range(n_components_95)])

print("\nPCA transformed data shape:", df_pca.shape)
print("\nSample of PCA transformed data:")
print(df_pca.head())

# %%
# Visualize first two principal components
plt.figure(figsize=(10, 8))
plt.scatter(X_pca_95[:, 0], X_pca_95[:, 1], alpha=0.3)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('First Two Principal Components')
plt.savefig('visualizations/pca_first_two_components.png')
plt.show()
plt.close()

# %%
# 4. Feature Importance based on PCA
components_df = pd.DataFrame(pca_95.components_.T, columns=[f'PC{i+1}' for i in range(n_components_95)], index=feature_names)
components_df['sum'] = components_df.abs().sum(axis=1)
top_features = components_df.sort_values('sum', ascending=False).head(20)

# %%
plt.figure(figsize=(12, 8))
sns.barplot(x='sum', y=top_features.index, data=top_features)
plt.title('Top 20 Features Importance based on PCA')
plt.xlabel('Absolute Sum of PCA Loadings')
plt.tight_layout()
plt.savefig('visualizations/pca_feature_importance.png')
plt.show()
plt.close()

# %%
print("\nTop 20 important features based on PCA:")
print(top_features['sum'])

# %%
# 5. Prepare data for modeling
X = df.drop('HadHeartAttack', axis=1)
y = df['HadHeartAttack']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use the preprocessor to transform the feature data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# %%
# 4. Implement Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_preprocessed, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test_preprocessed)
lr_probabilities = lr_model.predict_proba(X_test_preprocessed)[:, 1]

# Evaluate Logistic Regression
print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_predictions))

# %%
# 5. Implement Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_preprocessed, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_preprocessed)
rf_probabilities = rf_model.predict_proba(X_test_preprocessed)[:, 1]

# Evaluate Random Forest
print("\nRandom Forest Results:")
print(classification_report(y_test, rf_predictions))

# %%
# 6. Compare ROC curves
plt.figure(figsize=(10, 8))
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probabilities)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probabilities)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_probabilities):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probabilities):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('visualizations/roc_curve_comparison.png')
plt.show()
plt.close()



# %%
# 7. Feature Importance (for Random Forest)
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance (Random Forest)')# %% [Fixed Code]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
import os

# Create a directory to save visualizations
os.makedirs('visualizations', exist_ok=True)

# Load the data
df = pd.read_csv(r'C:\Users\iamaa\Downloads\dataset_heartdisease\2022\heart_2022_with_nans.csv')

# Separate features and target variable
X = df.drop('HadHeartAttack', axis=1)
y = df['HadHeartAttack']

# Identify categorical and numeric features
categorical_features = X.select_dtypes(include=['object']).columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

try:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
except TypeError:
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the preprocessor and transform the data
X_train_preprocessed = preprocessor.fit_transform(X_train)
X_test_preprocessed = preprocessor.transform(X_test)

# Get feature names
try:
    feature_names = (numeric_features.tolist() + 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names_out(categorical_features).tolist())
except AttributeError:
    feature_names = (numeric_features.tolist() + 
                     preprocessor.named_transformers_['cat']
                     .named_steps['onehot']
                     .get_feature_names(categorical_features).tolist())

# Convert to dense array if sparse
if hasattr(X_train_preprocessed, 'toarray'):
    X_train_preprocessed = X_train_preprocessed.toarray()
if hasattr(X_test_preprocessed, 'toarray'):
    X_test_preprocessed = X_test_preprocessed.toarray()

# Create DataFrames with preprocessed data
df_train_preprocessed = pd.DataFrame(X_train_preprocessed, columns=feature_names)
df_test_preprocessed = pd.DataFrame(X_test_preprocessed, columns=feature_names)

# Logistic Regression implementation after preprocessing
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train_preprocessed, y_train)

# Make predictions
lr_predictions = lr_model.predict(X_test_preprocessed)
lr_probabilities = lr_model.predict_proba(X_test_preprocessed)[:, 1]

# Evaluate Logistic Regression
print("\nLogistic Regression Results:")
print(classification_report(y_test, lr_predictions))

# Random Forest implementation
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_preprocessed, y_train)

# Make predictions
rf_predictions = rf_model.predict(X_test_preprocessed)
rf_probabilities = rf_model.predict_proba(X_test_preprocessed)[:, 1]

# Evaluate Random Forest
print("\nRandom Forest Results:")
print(classification_report(y_test, rf_predictions))

# Plot ROC Curve comparison
plt.figure(figsize=(10, 8))
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probabilities)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probabilities)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_score(y_test, lr_probabilities):.2f})')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_score(y_test, rf_probabilities):.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.savefig('visualizations/roc_curve_comparison.png')
plt.show()
plt.close()

# Feature importance for Random Forest
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance (Random Forest)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('visualizations/rf_feature_importance.png')
plt.show()
plt.close()

print("\nTop 20 important features based on Random Forest:")
print(feature_importance.head(20))

plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('visualizations/rf_feature_importance.png')
plt.show()
plt.close()

# %%
print("\nTop 20 important features based on Random Forest:")
print(feature_importance.head(20))


