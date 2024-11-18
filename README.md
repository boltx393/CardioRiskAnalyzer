# Cardio Risk Analyzer
Heart Disease Risk Assessment
Project for Semester - 7 

# Complete Heart Disease Analysis Pipeline

## 1. Data Ingestion & Initial Setup

```python
class HeartDiseaseAnalyzer:
    # Initialize with file paths and setup visualization manager
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.viz_manager = VisualizationManager()
        self.setup_models()
        self.setup_column_mappings()

```

## 2. Data Loading & Preprocessing

```python
def load_and_analyze_data():
    for file_path:
        1. Read CSV
        2. Standardize column names
        3. Encode categorical variables
        4. Calculate basic statistics
        5. Generate correlation matrices
        6. Calculate feature importance

def preprocess_data(df):
    1. Handle categorical variables
       - Binary encoding for Yes/No
       - Label encoding for others
    2. Handle numeric features
       - Impute missing values
       - Apply StandardScaler
    3. Return processed data and encoders

```

## 3. Feature Engineering & Dimensionality Reduction

```python
def apply_pca(X):
    1. Fit PCA (95% variance)
    2. Transform data
    3. Generate variance plots
    4. Return transformed data and fitted PCA

def analyze_feature_differences():
    1. Compare 2020 vs 2022 features
    2. Calculate correlation differences
    3. Identify unique features
    4. Generate comparison visualizations

```

## 4. Model Training & Evaluation

```python
def evaluate_model_with_cv(X, y, model):
    1. Create SMOTE pipeline
    2. Perform 3-fold CV
    3. For each fold:
        - Apply SMOTE
        - Train model
        - Generate predictions
        - Calculate metrics
    4. Return averaged metrics

def analyze_dataset(file_path):
    1. Preprocess data
    2. Apply PCA
    3. Evaluate all models
    4. Generate performance visualizations
    5. Return best model

```

## 5. Results Generation & Model Persistence

```python
def run_complete_analysis():
    1. Load and preprocess all datasets
    2. For each dataset:
        - Analyze features
        - Train and evaluate models
        - Select best model
        - Save model artifacts
    3. Generate comparative analysis
    4. Create visualizations
    5. Save analysis report

def save_analysis_report():
    1. Compile feature analysis
    2. Document model performance
    3. Save visualizations
    4. Export best models

```

## 6. Complete Pipeline Flow

```python
def main():
    1. Initialize analyzer
    2. Load datasets
    3. Preprocess data
    4. Engineer features
    5. Train models
    6. Generate results
    7. Save artifacts

Output Structure:
/visualizations_[timestamp]/
    /correlations/
    /distributions/
    /model_performance/
    /feature_analysis/
    /pca/
    analysis_report.txt
    best_model_[dataset].pkl

```