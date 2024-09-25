# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os

# %%
# Create a directory for saving visualizations
os.makedirs('visualizations', exist_ok=True)

# Function to save figures
def save_figure(fig, filename):
    fig.savefig(os.path.join('visualizations', filename))
    plt.close(fig)

# %%
# 1. Read the data
df = pd.read_csv(r'C:\Users\iamaa\OneDrive\Documents\GitHub\heartDiseaseRiskAssessment\heart_2022_with_nans.csv')

# %%
print("Data Shape:", df.shape)

# %%
print("\nData Info:")
df.info()

# %%
print("\nData Description:")
print(df.describe())

# %%
print("\nMissing Values:")
print(df.isnull().sum())

# %%
# Print column names and data types
print("\nColumn Names and Data Types:")
print(df.dtypes)

# %%
# Identify the target variable
target_variable = 'HadHeartAttack'
print(f"\nTarget Variable '{target_variable}' Data Type:", df[target_variable].dtype)
print(f"\nUnique values in '{target_variable}':", df[target_variable].unique())

# %%
# 2. Visualize original data
# Correlation heatmap for numeric features
numeric_features = df.select_dtypes(include=[np.number]).columns
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df[numeric_features].corr(), annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap of Numerical Features')
save_figure(fig, 'correlation_heatmap_original.png')
plt.show()

# %%
# Distribution of target variable
fig, ax = plt.subplots(figsize=(8, 6))
df[target_variable].value_counts().plot(kind='bar', ax=ax)
ax.set_title(f'Distribution of {target_variable} Cases')
ax.set_xlabel(target_variable)
ax.set_ylabel('Count')
save_figure(fig, 'target_distribution.png')
plt.show()

# %%
# 3. Preprocess the data
# Identify numeric and categorical columns
numeric_features = df.select_dtypes(include=[np.number]).columns
categorical_features = df.select_dtypes(include=['object']).columns

# Ensure target variable is not in feature lists
if target_variable in numeric_features:
    numeric_features = numeric_features.drop(target_variable)
if target_variable in categorical_features:
    categorical_features = categorical_features.drop(target_variable)

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Create a preprocessing pipeline
preprocessing_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Fit and transform the data
X = df.drop(target_variable, axis=1)
y = df[target_variable]

# Convert target variable to numeric if it's categorical
if y.dtype == 'object':
    y = pd.get_dummies(y, drop_first=True).iloc[:, 0]

X_preprocessed = preprocessing_pipeline.fit_transform(X)

# Get feature names after preprocessing
onehot_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = onehot_encoder.get_feature_names_out(categorical_features)
feature_names = list(numeric_features) + list(cat_feature_names)

# Convert to DataFrame for easier handling
df_preprocessed = pd.DataFrame(X_preprocessed.toarray(), columns=feature_names)
df_preprocessed[target_variable] = y

print("\nPreprocessed Data (first few rows):")
print(df_preprocessed.head())

# %%
# 4. Visualize preprocessed data
# Correlation heatmap of preprocessed data
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(df_preprocessed.corr(), annot=False, cmap='coolwarm', linewidths=0.5, ax=ax)
ax.set_title('Correlation Heatmap of Preprocessed Features')
save_figure(fig, 'correlation_heatmap_preprocessed.png')
plt.show()

# %%
# 5. Split the data
X_train, X_test, y_train, y_test = train_test_split(df_preprocessed.drop(target_variable, axis=1), df_preprocessed[target_variable], test_size=0.25, random_state=45)

# %%
# 6. Train and evaluate models
# Logistic Regression
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Results:")
print("Accuracy:", lr_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, lr_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, lr_pred))

# %%
# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Results:")
print("Accuracy:", rf_accuracy)
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# %%
# Compare feature importance (for Random Forest)
feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20), ax=ax)
ax.set_title('Top 20 Feature Importance (Random Forest)')
save_figure(fig, 'feature_importance.png')
plt.show()


