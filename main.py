import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pickle
import warnings
from tqdm import tqdm
import os

warnings.filterwarnings('ignore')

class VisualizationManager:
    def __init__(self):
        self.viz_dir = f'visualizations'
        self.create_directories()
        
    def create_directories(self):
        """Create necessary directories for visualizations."""
        subdirs = ['correlations', 'distributions', 'model_performance', 'feature_analysis', 'pca']
        os.makedirs(self.viz_dir, exist_ok=True)
        for subdir in subdirs:
            os.makedirs(os.path.join(self.viz_dir, subdir), exist_ok=True)
            
    def save_plot(self, subdir, filename):
        """Save current plot to appropriate subdirectory."""
        filepath = os.path.join(self.viz_dir, subdir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

class HeartDiseaseAnalyzer:
    def __init__(self, file_paths):
        self.file_paths = file_paths
        self.dataframes = {}
        self.analyses = {}
        self.viz_manager = VisualizationManager()
        self.models = {
            'Logistic Regression': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42),
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'XGBoost': XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5,  
                weights='uniform',
                algorithm='auto',
                leaf_size=30,
                p=2,  
                metric='minkowski',
                n_jobs=-1  
            )
        }
        
        # Define column mappings for different datasets
        self.column_mappings = {
            'heart_2020_cleaned.csv': {
                'target': 'HeartDisease',
                'features': {
                    'BMI': 'BMI',
                    'Smoking': 'SmokerStatus',
                    'AlcoholDrinking': 'AlcoholDrinkers',
                    'Stroke': 'HadStroke',
                    'PhysicalHealth': 'PhysicalHealthDays',
                    'MentalHealth': 'MentalHealthDays',
                    'DiffWalking': 'DifficultyWalking',
                    'Sex': 'Sex',
                    'AgeCategory': 'AgeCategory',
                    'Race': 'RaceEthnicityCategory',
                    'Diabetic': 'HadDiabetes',
                    'PhysicalActivity': 'PhysicalActivities',
                    'GenHealth': 'GeneralHealth',
                    'SleepTime': 'SleepHours',
                    'Asthma': 'HadAsthma',
                    'KidneyDisease': 'HadKidneyDisease',
                    'SkinCancer': 'HadSkinCancer'
                }
            },
            'heart_2022_no_nans.csv': {
                'target': 'HadHeartAttack',
                'features': None  # Use original column names
            },
            'heart_2022_with_nans.csv': {
                'target': 'HadHeartAttack',
                'features': None  # Use original column names
            }
        }
    
    def get_target_column(self, file_path):
        """Get the appropriate target column name for the dataset."""
        return self.column_mappings[file_path]['target']
    
    def standardize_columns(self, df, file_path):
        """Standardize column names across datasets."""
        mapping = self.column_mappings[file_path]['features']
        if mapping is not None:
            # Create a reverse mapping for analysis
            reverse_mapping = {v: k for k, v in mapping.items()}
            df_standardized = df.rename(columns=mapping)
            return df_standardized, reverse_mapping
        return df, None
    
    def encode_categorical_for_correlation(self, df):
        """Encode categorical variables for correlation analysis."""
        df_encoded = df.copy()
        
        # Encode target variable
        if 'HeartDisease' in df_encoded.columns:
            df_encoded['HeartDisease'] = df_encoded['HeartDisease'].map({'Yes': 1, 'No': 0})
        if 'HadHeartAttack' in df_encoded.columns:
            df_encoded['HadHeartAttack'] = df_encoded['HadHeartAttack'].map({'Yes': 1, 'No': 0})
            
        # Encode other categorical variables
        categorical_columns = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            # For binary Yes/No columns
            if set(df_encoded[col].unique()) == {'Yes', 'No'}:
                df_encoded[col] = df_encoded[col].map({'Yes': 1, 'No': 0})
            # For other categorical columns
            else:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def load_and_analyze_data(self):
        """Load and perform initial analysis on all datasets."""
        for file_path in self.file_paths:
            print(f"\nAnalyzing {file_path}...")
            
            # Load data
            df = pd.read_csv(file_path)
            print("\nOriginal columns:", df.columns.tolist())
            
            # Standardize column names
            df_standardized, reverse_mapping = self.standardize_columns(df, file_path)
            target_col = self.get_target_column(file_path)
            
            # Store both original and standardized dataframes
            self.dataframes[file_path] = {
                'original': df,
                'standardized': df_standardized,
                'reverse_mapping': reverse_mapping
            }
            
            # Encode categorical variables for correlation analysis
            df_encoded = self.encode_categorical_for_correlation(df)
            
            # Basic analysis
            analysis = {
                'shape': df.shape,
                'missing_values': df.isnull().sum(),
                'dtypes': df.dtypes,
                'basic_stats': df.describe(),
                'class_distribution': df[target_col].value_counts(normalize=True)
            }
            
            # Create correlation matrix using encoded data
            plt.figure(figsize=(15, 12))
            correlation_matrix = df_encoded.corr()
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
            plt.title(f'Correlation Matrix - {file_path}')
            plt.tight_layout()
            self.viz_manager.save_plot('correlations', f'correlation_matrix_{file_path.split("/")[-1].split(".")[0]}.png')
            
            # Feature importance based on correlation with target
            feature_importance = abs(correlation_matrix[target_col]).sort_values(ascending=False)
            
            # Save analysis results
            analysis['feature_importance'] = feature_importance
            self.analyses[file_path] = analysis
            
            print(f"\nDataset Shape: {analysis['shape']}")
            print(f"Missing Values:\n{analysis['missing_values'].sum()}")
            print(f"Class Distribution:\n{analysis['class_distribution']}")
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
    
    def preprocess_data(self, df):
        """Preprocess the data with appropriate cleaning and encoding."""
        df_processed = df.copy()
        label_encoders = {}
        
        # Get target column name
        target_col = None
        if 'HeartDisease' in df_processed.columns:
            target_col = 'HeartDisease'
        elif 'HadHeartAttack' in df_processed.columns:
            target_col = 'HadHeartAttack'
        
        # Handle categorical variables
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col == target_col:
                # Skip target column as it will be handled separately
                continue
            elif set(df_processed[col].unique()) == {'Yes', 'No'}:
                # Binary categorical variables
                df_processed[col] = df_processed[col].map({'Yes': 1, 'No': 0})
            else:
                # Other categorical variables
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
        
        # Handle numeric features
        numeric_features = df_processed.select_dtypes(include=['int64', 'float64']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Apply numeric transformer
        df_processed[numeric_features] = numeric_transformer.fit_transform(df_processed[numeric_features])
        
        return df_processed, label_encoders

    def apply_pca(self, X, n_components=0.95):
        """Apply PCA and visualize results."""
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)
        
        # Plot explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(cumulative_variance_ratio) + 1), cumulative_variance_ratio, 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('PCA - Explained Variance Ratio')
        plt.grid(True)
        self.viz_manager.save_plot('pca', 'pca_explained_variance.png')
        
        print(f"\nNumber of components selected: {X_pca.shape[1]}")
        print(f"Total explained variance ratio: {cumulative_variance_ratio[-1]:.4f}")
        
        return X_pca, pca
    
    def create_evaluation_pipeline(self, model):
        """Create a pipeline with SMOTE and the model."""
        return ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', model)
        ])
    
    def evaluate_model_with_cv(self, X, y, model_name, model):
        """Evaluate model using 5-fold cross-validation with SMOTE."""
        pipeline = self.create_evaluation_pipeline(model)
        kfold = KFold(n_splits=3, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            print(f"\nFold {fold}:")
            print(f"Training set class distribution: {np.bincount(y_train)}")
            print(f"Validation set class distribution: {np.bincount(y_val)}")
            
            # Fit pipeline and predict
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_val)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            try:
                y_prob = pipeline.predict_proba(X_val)[:, 1]
                auc_roc = roc_auc_score(y_val, y_prob)
            except:
                auc_roc = None
            
            fold_metrics.append({
                'fold': fold,
                'accuracy': accuracy,
                'auc_roc': auc_roc,
                'confusion_matrix': confusion_matrix(y_val, y_pred),
                'classification_report': classification_report(y_val, y_pred)
            })
            
            print(f"Fold {fold} Accuracy: {accuracy:.4f}")
            if auc_roc:
                print(f"Fold {fold} AUC-ROC: {auc_roc:.4f}")
        
        return fold_metrics
    
    def analyze_dataset(self, file_path):
        """Analyze a single dataset completely."""
        print(f"\nAnalyzing dataset: {file_path}")
        
        # Get the data
        df = self.dataframes[file_path]['standardized']
        target_col = self.get_target_column(file_path)
        
        # First ensure target is binary
        if target_col == 'HeartDisease':
            y = (df[target_col] == 'Yes').astype(int)
        elif target_col == 'HadHeartAttack':
            y = (df[target_col] == 'Yes').astype(int)
        else:
            y = df[target_col]
            
        # Then preprocess features
        X = df.drop(target_col, axis=1)
        df_processed, label_encoders = self.preprocess_data(pd.concat([X, pd.Series(y, name=target_col)], axis=1))
        X = df_processed.drop(target_col, axis=1)
        
        # Apply PCA
        X_pca, pca = self.apply_pca(X)
        X_pca = np.array(X_pca)
        y = np.array(y)
        
        print("\nClass distribution in target:")
        print(pd.Series(y).value_counts(normalize=True))
        
        # Evaluate each model
        model_results = {}
        
        for model_name, model in tqdm(self.models.items(), desc="Evaluating models"):
            print(f"\nEvaluating {model_name}")
            fold_metrics = self.evaluate_model_with_cv(X_pca, y, model_name, model)
            
            # Calculate average metrics
            avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
            avg_auc_roc = np.mean([m['auc_roc'] for m in fold_metrics if m['auc_roc'] is not None])
            
            model_results[model_name] = {
                'fold_metrics': fold_metrics,
                'avg_accuracy': avg_accuracy,
                'avg_auc_roc': avg_auc_roc
            }
            
            print(f"Average Accuracy: {avg_accuracy:.4f}")
            if avg_auc_roc:
                print(f"Average AUC-ROC: {avg_auc_roc:.4f}")
        
        return model_results, pca, label_encoders

    def run_complete_analysis(self):
        """Run complete analysis on all datasets."""
        # First load and analyze all datasets
        self.load_and_analyze_data()
        
        # Analyze each dataset
        all_results = {}
        for file_path in self.file_paths:
            print(f"\n{'='*50}")
            print(f"Processing {file_path}")
            print(f"{'='*50}")
            
            model_results, pca, label_encoders = self.analyze_dataset(file_path)
            
            # Find best model for this dataset
            best_model_name = max(model_results.items(), 
                                key=lambda x: x[1]['avg_accuracy'])[0]
            
            all_results[file_path] = {
                'model_results': model_results,
                'best_model': best_model_name,
                'pca': pca,
                'label_encoders': label_encoders
            }
            
            # Save best model for this dataset
            best_model_data = {
                'model': self.models[best_model_name],
                'pca': pca,
                'label_encoders': label_encoders
            }
            
            filename = f'best_model_{file_path.split("/")[-1].split(".")[0]}.pkl'
            with open(filename, 'wb') as f:
                pickle.dump(best_model_data, f)
            
            # Create visualizations for this dataset
            self.visualize_distributions(self.dataframes[file_path]['standardized'], 
                                      file_path.split('.')[0])
            self.visualize_model_performance(all_results[file_path], 
                                          file_path.split('.')[0])
            
            print(f"\nBest model for {file_path}: {best_model_name}")
            print(f"Average Accuracy: {model_results[best_model_name]['avg_accuracy']:.4f}")
            if model_results[best_model_name]['avg_auc_roc']:
                print(f"Average AUC-ROC: {model_results[best_model_name]['avg_auc_roc']:.4f}")
        
        # Analyze feature differences between 2020 and 2022 datasets
        feature_differences = self.analyze_feature_differences()
        
        # Save comprehensive analysis report
        self.save_analysis_report(feature_differences, all_results)
        
        return all_results

    def visualize_distributions(self, df, dataset_name):
        """Visualize distributions of numerical features."""
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        target_col = self.get_target_column(dataset_name + '.csv')
        
        for col in numerical_cols:
            if col != target_col:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=df, x=col, hue=target_col, multiple="stack")
                plt.title(f'Distribution of {col} by Heart Disease Status - {dataset_name}')
                plt.xticks(rotation=45)
                self.viz_manager.save_plot('distributions', f'dist_{dataset_name}_{col}.png')

    def visualize_model_performance(self, results, dataset_name):
        """Visualize model performance metrics."""
        # Accuracy Comparison
        plt.figure(figsize=(10, 6))
        accuracies = [metrics['avg_accuracy'] for metrics in results['model_results'].values()]
        model_names = list(results['model_results'].keys())
        
        plt.bar(model_names, accuracies)
        plt.title(f'Model Accuracy Comparison - {dataset_name}')
        plt.xticks(rotation=45)
        plt.ylabel('Average Accuracy')
        self.viz_manager.save_plot('model_performance', f'accuracy_comparison_{dataset_name}.png')
        
        # ROC-AUC Comparison
        plt.figure(figsize=(10, 6))
        aucs = [metrics['avg_auc_roc'] for metrics in results['model_results'].values()]
        
        plt.bar(model_names, aucs)
        plt.title(f'Model ROC-AUC Comparison - {dataset_name}')
        plt.xticks(rotation=45)
        plt.ylabel('Average ROC-AUC')
        self.viz_manager.save_plot('model_performance', f'roc_auc_comparison_{dataset_name}.png')

    def analyze_feature_differences(self):
        """Analyze differences between 2020 and 2022 datasets."""
        df_2020 = self.dataframes['heart_2020_cleaned.csv']['standardized']
        df_2022 = self.dataframes['heart_2022_no_nans.csv']['standardized']
        
        target_2020 = self.get_target_column('heart_2020_cleaned.csv')
        target_2022 = self.get_target_column('heart_2022_no_nans.csv')
        
        # Compare features
        features_2020 = set(df_2020.columns) - {target_2020}
        features_2022 = set(df_2022.columns) - {target_2022}
        
        unique_2020 = features_2020 - features_2022
        unique_2022 = features_2022 - features_2020
        common_features = features_2020.intersection(features_2022)
        
        # Analyze impact of common features
        feature_impact = {}
        for feature in common_features:
            if feature not in [target_2020, target_2022]:
                # Convert target variables to numeric for correlation
                target_2020_numeric = (df_2020[target_2020] == 'Yes').astype(int)
                target_2022_numeric = (df_2022[target_2022] == 'Yes').astype(int)
                
                # Calculate correlation with target for both datasets
                if df_2020[feature].dtype == 'object':
                    feature_2020 = pd.get_dummies(df_2020[feature]).iloc[:, 0]
                else:
                    feature_2020 = df_2020[feature]
                    
                if df_2022[feature].dtype == 'object':
                    feature_2022 = pd.get_dummies(df_2022[feature]).iloc[:, 0]
                else:
                    feature_2022 = df_2022[feature]
                
                corr_2020 = feature_2020.corr(target_2020_numeric)
                corr_2022 = feature_2022.corr(target_2022_numeric)
                
                feature_impact[feature] = {
                    'correlation_2020': corr_2020,
                    'correlation_2022': corr_2022,
                    'correlation_diff': abs(corr_2020 - corr_2022)
                }
        
        # Visualize feature differences
        self.visualize_feature_differences(feature_impact)
        
        return {
            'unique_2020': unique_2020,
            'unique_2022': unique_2022,
            'common_features': common_features,
            'feature_impact': feature_impact
        }

    def visualize_feature_differences(self, feature_impact):
        """Create visualizations for feature differences."""
        # 1. Correlation Difference Plot
        plt.figure(figsize=(15, 8))
        features = list(feature_impact.keys())
        corr_diff = [abs(impact['correlation_diff']) for impact in feature_impact.values()]
        
        plt.bar(features, corr_diff)
        plt.xticks(rotation=45, ha='right')
        plt.title('Absolute Correlation Difference Between 2020 and 2022 Datasets')
        plt.ylabel('Absolute Correlation Difference')
        self.viz_manager.save_plot('feature_analysis', 'correlation_differences.png')
        
        # 2. Feature Correlation Comparison
        plt.figure(figsize=(15, 8))
        x = np.arange(len(features))
        width = 0.35
        
        corr_2020 = [impact['correlation_2020'] for impact in feature_impact.values()]
        corr_2022 = [impact['correlation_2022'] for impact in feature_impact.values()]
        
        plt.bar(x - width/2, corr_2020, width, label='2020')
        plt.bar(x + width/2, corr_2022, width, label='2022')
        
        plt.xlabel('Features')
        plt.ylabel('Correlation with Heart Disease')
        plt.title('Feature Correlation Comparison: 2020 vs 2022')
        plt.xticks(x, features, rotation=45, ha='right')
        plt.legend()
        self.viz_manager.save_plot('feature_analysis', 'correlation_comparison.png')

    def save_analysis_report(self, feature_differences, all_results):
        """Save comprehensive analysis report."""
        report = []
        
        # Feature Differences Section
        report.append("FEATURE ANALYSIS BETWEEN 2020 AND 2022 DATASETS")
        report.append("=" * 50)
        report.append("\nUnique Features in 2020:")
        report.extend([f"- {feature}" for feature in feature_differences['unique_2020']])
        
        report.append("\nUnique Features in 2022:")
        report.extend([f"- {feature}" for feature in feature_differences['unique_2022']])
        
        report.append("\nMost Impactful Common Features (by correlation difference):")
        sorted_features = sorted(
            feature_differences['feature_impact'].items(),
            key=lambda x: abs(x[1]['correlation_diff']),
            reverse=True
        )
        for feature, impact in sorted_features[:10]:
            report.append(f"\n{feature}:")
            report.append(f"  2020 correlation: {impact['correlation_2020']:.4f}")
            report.append(f"  2022 correlation: {impact['correlation_2022']:.4f}")
            report.append(f"  Absolute difference: {impact['correlation_diff']:.4f}")
        
        # Model Performance Section
        report.append("\n\nMODEL PERFORMANCE COMPARISON")
        report.append("=" * 50)
        
        for dataset, results in all_results.items():
            report.append(f"\n{dataset}:")
            report.append("-" * 30)
            for model_name, metrics in results['model_results'].items():
                report.append(f"\n{model_name}:")
                report.append(f"  Average Accuracy: {metrics['avg_accuracy']:.4f}")
                if metrics['avg_auc_roc']:
                    report.append(f"  Average AUC-ROC: {metrics['avg_auc_roc']:.4f}")
            
            report.append(f"\nBest Model: {results['best_model']}")
        
        # Save report
        report_path = os.path.join(self.viz_manager.viz_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))

def main():
    # Define file paths
    file_paths = [
        'heart_2020_cleaned.csv',
        'heart_2022_no_nans.csv',
        'heart_2022_with_nans.csv'
    ]
    
    # Create analyzer instance
    analyzer = HeartDiseaseAnalyzer(file_paths)
    
    # Run complete analysis
    all_results = analyzer.run_complete_analysis()
    
    print(f"\nAnalysis complete. Results saved in: {analyzer.viz_manager.viz_dir}")

if __name__ == "__main__":
    main()