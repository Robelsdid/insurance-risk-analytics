"""
Data Loader Module for Insurance Risk Analytics

This module handles loading and basic preprocessing of insurance data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsuranceDataLoader:
    """
    A class to load and preprocess insurance data for risk analytics.
    """
    
    def __init__(self, data_path: str = "data/MachineLearningRating_v3.txt"):
        """
        Initialize the data loader.
        
        Args:
            data_path (str): Path to the insurance data file
        """
        self.data_path = Path(data_path)
        self.data: Optional[pd.DataFrame] = None
        self.raw_data: Optional[pd.DataFrame] = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the insurance data from the specified file.
        
        Returns:
            pd.DataFrame: Loaded insurance data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            if not self.data_path.exists():
                raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
            # Load the data - assuming it's tab-separated based on the .txt extension
            self.raw_data = pd.read_csv(self.data_path, sep='|')
            parse_dates = ['TransactionMonth']
            self.data = self.raw_data.copy()
            
            logger.info(f"Successfully loaded {len(self.data)} records with {len(self.data.columns)} columns")
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Get basic information about the loaded data.
        
        Returns:
            Dict containing data shape, column info, and basic stats
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        info = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
        
        return info
    
    def get_sample_data(self, n: int = 5) -> pd.DataFrame:
        """
        Get a sample of the data for inspection.
        
        Args:
            n (int): Number of sample rows to return
            
        Returns:
            pd.DataFrame: Sample data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        return self.data.head(n)
    
    def reset_data(self):
        """Reset data to the original loaded state."""
        if self.raw_data is not None:
            self.data = self.raw_data.copy()
            logger.info("Data reset to original state")
        else:
            logger.warning("No original data to reset to")


if __name__ == "__main__":
    # Example usage
    loader = InsuranceDataLoader()
    data = loader.load_data()
    print(loader.get_data_info())
    print("\nSample data:")
    print(loader.get_sample_data())
