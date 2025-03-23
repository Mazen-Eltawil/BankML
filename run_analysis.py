"""
Script to run the Bank Marketing Campaign Analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import argparse

from bank_marketing import (
    ModelConfig,
    DataLoader,
    DataPreprocessor,
    ModelTrainer,
    DataLoadingError,
    PreprocessingError,
    ModelTrainingError,
    ValidationError,
)
from bank_marketing.constants import DEFAULT_PATHS

def setup_logging(log_dir: str = 'logs') -> None:
    """Set up logging configuration."""
    Path(log_dir).mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(Path(log_dir) / 'analysis.log')
        ]
    )

def main(config_path: Optional[str] = None, data_path: Optional[str] = None) -> None:
    """
    Main function to run the analysis pipeline.
    
    Args:
        config_path: Path to the configuration file
        data_path: Path to the data file (overrides config file)
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize configuration
        config = ModelConfig(config_path=config_path)
        if data_path:
            config.data_path = data_path
            
        # Create directories
        for dir_path in DEFAULT_PATHS.values():
            Path(dir_path).mkdir(exist_ok=True)
            
        # Log configuration
        logger.info(f"Using configuration: {config}")
            
        # Load data
        logger.info("Starting data loading...")
        data_loader = DataLoader(config)
        data = data_loader.load_data()
        
        # Preprocess data
        logger.info("Starting data preprocessing...")
        preprocessor = DataPreprocessor(config)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.preprocess(data)
        
        # Get feature names
        feature_names = preprocessor.feature_names
        if feature_names is None:
            raise ValidationError("Feature names not properly set during preprocessing")
        logger.info(f"Using features: {feature_names}")
        
        # Train and evaluate models
        logger.info("Starting model training...")
        trainer = ModelTrainer(config, feature_names=feature_names)
        trainer.train_models(X_train, y_train, X_val, y_val)
        
        logger.info("Analysis pipeline completed successfully!")
        
    except (DataLoadingError, PreprocessingError, ModelTrainingError, ValidationError) as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Bank Marketing Analysis")
    parser.add_argument("--config", dest="config_path", required=True, help="Path to config file")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the input data file (overrides config)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for log files"
    )
    
    args = parser.parse_args()
    setup_logging(args.log_dir)
    try:
        main(args.config_path, args.data_path)
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}")
        raise 