"""
Bank Marketing Campaign Analysis
Advanced ML Pipeline with Error Handling and Optimization

Author: [Your Name]
Date: [Current Date]
"""

def check_and_install_packages():
    """Check if required packages are installed and install only missing ones."""
    import importlib.util
    import subprocess
    import sys

    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'imblearn': 'imbalanced-learn',
        'requests': 'requests',
        'zipfile': 'zipfile36'
    }

    missing_packages = []
    
    for package, pip_name in required_packages.items():
        if importlib.util.find_spec(package) is None:
            missing_packages.append(pip_name)

    if missing_packages:
        print("Installing missing packages:", missing_packages)
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Successfully installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install {package}: {str(e)}")
        print("Package installation completed!")
    else:
        print("All required packages are already installed!")

# Check and install required packages
check_and_install_packages()

import os
import logging
import warnings
import gc
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm

# Core data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create directories for artifacts
ARTIFACT_DIRS = ['models', 'plots', 'reports']
for dir_name in ARTIFACT_DIRS:
    Path(dir_name).mkdir(parents=True, exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def log_memory_usage(logger, operation: str):
    """Log memory usage before and after an operation."""
    memory_usage = get_memory_usage()
    logger.info(f"Memory usage {operation}: {memory_usage:.2f} MB")

@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2
    cv_folds: int = 3
    n_jobs: int = -1  # Use all CPU cores
    batch_size: int = 1000  # Batch size for processing large datasets

class BankMarketingDataset:
    """Handles data loading, preprocessing, and feature engineering."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data: Optional[pd.DataFrame] = None
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None
        self.feature_names: Optional[List[str]] = None
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.numeric_imputer = SimpleImputer(strategy='mean')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.logger = logging.getLogger(__name__)
        
    def load_data(self) -> None:
        """Load data from local file or URL."""
        try:
            # Try loading from local file first
            file_path = Path("bank-additional-full.csv")
            if file_path.exists():
                logger.info(f"Loading data from local file: {file_path}")
                self.data = pd.read_csv(file_path, sep=';')
            else:
                # If local file not found, try downloading from URL
                logger.info("Local file not found. Downloading from URL...")
                import requests
                url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
                from zipfile import ZipFile
                from io import BytesIO
                
                response = requests.get(url)
                zipfile = ZipFile(BytesIO(response.content))
                self.data = pd.read_csv(zipfile.open('bank-additional/bank-additional-full.csv'), sep=';')
                
                # Save to local file for future use
                self.data.to_csv(file_path, sep=';', index=False)
                logger.info(f"Data saved to local file: {file_path}")
            
            logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            self._validate_data()
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _validate_data(self) -> None:
        """Validate data quality and completeness."""
        if self.data is None:
            raise ValueError("Data not loaded")
        
        # Check for missing values
        missing = self.data.isnull().sum()
        if missing.any():
            logger.warning(f"Missing values found:\n{missing[missing > 0]}")
        
        # Log target variable info
        logger.info("\nTarget variable distribution:")
        logger.info(self.data['y'].value_counts(normalize=True))

    def validate_data_quality(self):
        """Validate data quality and completeness."""
        if self.data is None:
            raise ValueError("Data not loaded")
            
        # Check for missing values
        missing_values = self.data.isnull().sum()
        if missing_values.any():
            self.logger.warning(f"Missing values detected:\n{missing_values[missing_values > 0]}")
        
        # Check for data types
        self.logger.info(f"Data types:\n{self.data.dtypes}")
        
        # Check for duplicates
        duplicates = self.data.duplicated().sum()
        if duplicates > 0:
            self.logger.warning(f"Found {duplicates} duplicate rows")

    def perform_eda(self) -> None:
        """Perform exploratory data analysis."""
        try:
            logger.info("Starting Exploratory Data Analysis...")
            
            # Create plots directory if it doesn't exist
            plots_dir = Path('plots')
            plots_dir.mkdir(exist_ok=True)
            
            # 1. Analyze numerical distributions
            self._plot_numerical_distributions()
            
            # 2. Analyze categorical variables
            self._plot_categorical_distributions()
            
            # 3. Analyze correlations
            self._plot_correlation_matrix()
            
            # 4. Analyze target relationship with key features
            self._plot_target_relationships()
            
            logger.info("EDA completed successfully")
            
        except Exception as e:
            logger.error(f"Error in EDA: {str(e)}")
            raise

    def _plot_numerical_distributions(self) -> None:
        """Plot distributions of numerical features."""
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'y']
        
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            # Convert to numeric if possible
            try:
                self.data[col] = pd.to_numeric(self.data[col])
            except (ValueError, TypeError):
                pass
            sns.histplot(data=self.data, x=col, hue='y', multiple="stack")
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/numerical_distributions.png')
        plt.close()

    def _plot_categorical_distributions(self) -> None:
        """Plot distributions of categorical features."""
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'y']
        
        n_cols = 2
        n_rows = (len(categorical_cols) + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, col in enumerate(categorical_cols, 1):
            plt.subplot(n_rows, n_cols, i)
            # Convert categorical to string type to avoid numeric parsing warning
            self.data[col] = self.data[col].astype(str)
            sns.countplot(data=self.data, x=col, hue='y')
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('plots/categorical_distributions.png')
        plt.close()

    def _plot_correlation_matrix(self) -> None:
        """Plot correlation matrix for numerical features."""
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        correlation_matrix = self.data[numerical_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix of Numerical Features')
        plt.tight_layout()
        plt.savefig('plots/correlation_matrix.png')
        plt.close()

    def _plot_target_relationships(self) -> None:
        """Plot relationships between features and target variable."""
        # Select top numerical features
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        numerical_cols = [col for col in numerical_cols if col != 'y'][:6]  # Top 6 features
        
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(numerical_cols, 1):
            plt.subplot(2, 3, i)
            sns.boxplot(data=self.data, x='y', y=col)
            plt.title(f'{col} by Target')
        
        plt.tight_layout()
        plt.savefig('plots/target_relationships.png')
        plt.close()

    def _process_batch(self, batch: pd.DataFrame) -> pd.DataFrame:
        """Process a single batch of data."""
        try:
            # Handle numerical variables
            numerical_cols = batch.select_dtypes(include=['int64', 'float64']).columns
            numerical_cols = [col for col in numerical_cols if col != 'y']
            
            # Handle categorical variables
            categorical_cols = batch.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if col != 'y']
            
            # Impute missing values first
            if numerical_cols:
                batch[numerical_cols] = pd.DataFrame(
                    self.numeric_imputer.fit_transform(batch[numerical_cols]),
                    columns=numerical_cols,
                    index=batch.index
                )
            if categorical_cols:
                batch[categorical_cols] = pd.DataFrame(
                    self.categorical_imputer.fit_transform(batch[categorical_cols]),
                    columns=categorical_cols,
                    index=batch.index
                )
            
            # One-hot encode categorical variables
            batch = pd.get_dummies(batch, columns=categorical_cols)
            
            # Verify no NaN values remain
            if batch.isna().any().any():
                raise ValueError("NaN values found after imputation")
                
            return batch
            
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise

    def preprocess(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                 np.ndarray, np.ndarray, np.ndarray]:
        """Preprocess the data for model training."""
        try:
            logger.info("Starting preprocessing...")
            log_memory_usage(logger, "before preprocessing")
            
            # Process data in batches for large datasets
            batch_size = self.config.batch_size
            n_samples = len(self.data)
            
            if n_samples > batch_size:
                logger.info(f"Processing {n_samples} samples in batches of {batch_size}")
                processed_data = []
                
                for i in tqdm(range(0, n_samples, batch_size), desc="Processing batches"):
                    batch = self.data.iloc[i:i+batch_size].copy()
                    # Process batch
                    batch = self._process_batch(batch)
                    processed_data.append(batch)
                    
                    # Clear memory
                    del batch
                    gc.collect()
                
                # Combine processed batches
                data = pd.concat(processed_data, axis=0)
                del processed_data
                gc.collect()
            else:
                data = self._process_batch(self.data.copy())
            
            # Prepare features and target
            X = data.drop('y', axis=1)
            y = (data['y'] == 'yes').astype(int)
            
            # Store feature names
            self.feature_names = X.columns.tolist()
            
            # Scale features
            X = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            # Handle any remaining NaN values after scaling
            if X.isna().any().any():
                logger.warning("Handling remaining NaN values after scaling")
                X = X.fillna(X.mean())
            
            # Split data
            X_temp, self.X_test, y_temp, self.y_test = train_test_split(
                X, y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y
            )
            
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
                X_temp, y_temp,
                test_size=self.config.validation_size,
                random_state=self.config.random_state,
                stratify=y_temp
            )
            
            # Convert to numpy arrays and ensure no NaN values
            self.X_train = self.X_train.fillna(self.X_train.mean()).values
            self.X_val = self.X_val.fillna(self.X_val.mean()).values
            self.X_test = self.X_test.fillna(self.X_test.mean()).values
            
            # Verify no NaN values before SMOTE
            if np.isnan(self.X_train).any():
                raise ValueError("NaN values found before SMOTE")
            
            # Apply SMOTE to training data only
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.config.random_state)
            self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
            
            log_memory_usage(logger, "after preprocessing")
            
            logger.info(f"Training set shape: {self.X_train.shape}")
            logger.info(f"Validation set shape: {self.X_val.shape}")
            logger.info(f"Test set shape: {self.X_test.shape}")
            
            return (self.X_train, self.X_val, self.X_test,
                    self.y_train, self.y_val, self.y_test)
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise

class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models: Dict[str, object] = {}
        self.results: Dict[str, Dict] = {}
        
    def train_models(self, dataset: BankMarketingDataset) -> None:
        """Train multiple models and evaluate their performance."""
        models_to_train = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=self.config.random_state
            )
        }
        
        logger.info("Starting model training...")
        log_memory_usage(logger, "before model training")
        
        for name, model in tqdm(models_to_train.items(), desc="Training models"):
            logger.info(f"\nTraining {name}...")
            self._train_and_evaluate_model(name, model, dataset)
        
        log_memory_usage(logger, "after model training")
        
        # Generate analysis report
        self._generate_report(dataset)
        
    def _train_and_evaluate_model(self, name: str, model: object,
                                dataset: BankMarketingDataset) -> None:
        """Train and evaluate a single model."""
        try:
            # Train model with progress bar for iterations
            if hasattr(model, 'n_estimators'):
                with tqdm(total=model.n_estimators, desc=f"Training {name}") as pbar:
                    original_fit = model.fit
                    def fit_with_progress(*args, **kwargs):
                        result = original_fit(*args, **kwargs)
                        pbar.update(1)
                        return result
                    model.fit = fit_with_progress
                    model.fit(dataset.X_train, dataset.y_train)
                    model.fit = original_fit
            else:
                model.fit(dataset.X_train, dataset.y_train)
            
            self.models[name] = model
            
            # Get predictions with progress bar
            logger.info("Generating predictions...")
            y_pred = model.predict(dataset.X_test)
            y_pred_proba = model.predict_proba(dataset.X_test)[:, 1]
            
            # Calculate metrics
            self.results[name] = {
                'accuracy': accuracy_score(dataset.y_test, y_pred),
                'precision': precision_score(dataset.y_test, y_pred),
                'recall': recall_score(dataset.y_test, y_pred),
                'f1': f1_score(dataset.y_test, y_pred),
                'roc_auc': roc_auc_score(dataset.y_test, y_pred_proba)
            }
            
            # Plot visualizations with progress bar
            with tqdm(total=3, desc=f"Generating {name} plots") as pbar:
                self._plot_roc_curve(dataset.y_test, y_pred_proba, name)
                pbar.update(1)
                self._plot_confusion_matrix(dataset.y_test, y_pred, name)
                pbar.update(1)
                if hasattr(model, 'feature_importances_'):
                    self._plot_feature_importance(model, dataset.feature_names, name)
                pbar.update(1)
            
            logger.info(f"\n{name} Results:")
            for metric, value in self.results[name].items():
                logger.info(f"{metric}: {value:.4f}")
            
        except Exception as e:
            logger.error(f"Error training {name}: {str(e)}")
            raise

    def _plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                       model_name: str) -> None:
        """Plot ROC curve for a model."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.savefig(f'plots/{model_name}_roc_curve.png')
        plt.close()

    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'plots/{model_name}_confusion_matrix.png')
        plt.close()

    def _plot_feature_importance(self, model: object, feature_names: List[str],
                               model_name: str) -> None:
        """Plot feature importance."""
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance.head(20), x='importance', y='feature')
        plt.title(f'Top 20 Feature Importance - {model_name}')
        plt.tight_layout()
        plt.savefig(f'plots/{model_name}_feature_importance.png')
        plt.close()

    def _generate_report(self, dataset: BankMarketingDataset) -> None:
        """Generate analysis report in HTML format."""
        try:
            report_content = [
                "<!DOCTYPE html>",
                "<html><head>",
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 40px; }",
                "h1, h2 { color: #2c3e50; }",
                "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
                "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                "th { background-color: #f5f5f5; }",
                ".metric-good { color: green; }",
                ".metric-warning { color: orange; }",
                ".metric-bad { color: red; }",
                "</style>",
                "</head><body>",
                "<h1>Bank Marketing Campaign Analysis Report</h1>",
                
                "<h2>1. Dataset Overview</h2>",
                f"<p>Total samples: {len(dataset.data):,}</p>",
                "<h3>Class Distribution</h3>",
                f"<p>Positive class (Subscribed): {(dataset.data['y'] == 'yes').sum():,} "
                f"({(dataset.data['y'] == 'yes').mean()*100:.1f}%)</p>",
                
                "<h2>2. Model Performance</h2>",
                "<table>",
                "<tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>ROC AUC</th></tr>"
            ]
            
            # Add model performance rows
            for model_name, metrics in self.results.items():
                row = "<tr>"
                row += f"<td>{model_name}</td>"
                for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                    value = metrics[metric]
                    color_class = 'metric-good' if value > 0.7 else ('metric-warning' if value > 0.5 else 'metric-bad')
                    row += f"<td class='{color_class}'>{value:.3f}</td>"
                row += "</tr>"
                report_content.append(row)
            
            report_content.extend([
                "</table>",
                "<h2>3. Visualizations</h2>",
                "<p>Please refer to the 'plots' directory for detailed visualizations including:</p>",
                "<ul>",
                "<li>ROC curves</li>",
                "<li>Confusion matrices</li>",
                "<li>Feature importance plots</li>",
                "<li>Distribution plots</li>",
                "</ul>",
                "</body></html>"
            ])
            
            # Save report
            with open('reports/analysis_report.html', 'w') as f:
                f.write("\n".join(report_content))
            
            logger.info("Analysis report generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise

def main():
    """Main execution function."""
    try:
        # Initialize configuration and dataset
        config = ModelConfig()
        dataset = BankMarketingDataset(config)
        
        # Load and analyze data
        dataset.load_data()
        dataset.perform_eda()
        
        # Preprocess data
        dataset.preprocess()
        
        # Train and evaluate models
        model_trainer = ModelTrainer(config)
        model_trainer.train_models(dataset)
        
        # Clean up memory
        gc.collect()
        
        logger.info("Analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 