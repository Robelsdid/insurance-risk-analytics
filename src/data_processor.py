import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class for processing insurance claim data."""
    
    def __init__(self, data_dir: str = '../data'):
        """Initialize the DataProcessor.
        
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = data_dir
        self.raw_data = None
        self.processed_data = None
        
    def load_data(self, filename: str, parse_dates: List[str] = None, sep: str = ',') -> pd.DataFrame:
        """Load data from a CSV file.
        
        Args:
            filename: Name of the CSV file to load
            parse_dates: List of column names to parse as dates
            sep: Delimiter to use for parsing the file (default: ',')
            
        Returns:
            Loaded DataFrame
        """
        file_path = os.path.join(self.data_dir, filename)
        logger.info(f"Loading data from {file_path}")
        
        # Default date columns to parse if none provided
        if parse_dates is None:
            parse_dates = ['transaction_date', 'policy_start_date', 'policy_end_date', 'claim_date']
        
        try:
            data = pd.read_csv(file_path, parse_dates=parse_dates, sep=sep)
            logger.info(f"Successfully loaded data with shape {data.shape}")
            
            # Log basic information about the loaded data
            if 'transaction_date' in data.columns:
                logger.info(f"Data timeframe: {data['transaction_date'].min()} to {data['transaction_date'].max()}")
            if 'policy_id' in data.columns:
                logger.info(f"Number of unique policies: {data['policy_id'].nunique()}")
            if 'client_id' in data.columns:
                logger.info(f"Number of unique clients: {data['client_id'].nunique()}")
            
            self.raw_data = data
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def clean_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Clean the data by handling missing values, outliers, etc.
        
        Args:
            data: DataFrame to clean, if None, uses self.raw_data
            
        Returns:
            Cleaned DataFrame
        """
        if data is None:
            if self.raw_data is None:
                raise ValueError("No data loaded. Call load_data first.")
            data = self.raw_data.copy()
        
        logger.info("Cleaning data...")
        
        # Handle missing values
        missing_counts = data.isnull().sum()
        logger.info(f"Missing value counts:\n{missing_counts[missing_counts > 0]}")
        
        # Convert date columns to datetime
        date_columns = ['transaction_date', 'policy_start_date', 'policy_end_date', 'claim_date']
        for col in date_columns:
            if col in data.columns:
                data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # For numeric columns, fill missing values with median
        numeric_cols = data.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].median())
        
        # For categorical columns, fill missing values with mode
        cat_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            if col not in date_columns and data[col].isnull().sum() > 0:
                data[col] = data[col].fillna(data[col].mode()[0])
        
        # Handle specific columns based on domain knowledge
        # Fill missing claim amounts with 0 (assuming no claim)
        if 'claim_amount' in data.columns and data['claim_amount'].isnull().sum() > 0:
            data['claim_amount'] = data['claim_amount'].fillna(0)
        
        # Fill missing car value with median of same car model
        if 'car_value' in data.columns and 'car_model' in data.columns and data['car_value'].isnull().sum() > 0:
            car_model_medians = data.groupby('car_model')['car_value'].median().to_dict()
            for model in car_model_medians:
                model_mask = (data['car_model'] == model) & (data['car_value'].isnull())
                data.loc[model_mask, 'car_value'] = car_model_medians[model]
            # If any still missing, use overall median
            if data['car_value'].isnull().sum() > 0:
                data['car_value'] = data['car_value'].fillna(data['car_value'].median())
        
        # Handle outliers using IQR method for specific numeric columns
        # Only apply to financial and risk-related columns
        outlier_cols = [col for col in numeric_cols if any(x in col.lower() for x in 
                                                         ['amount', 'value', 'premium', 'claim', 'coverage', 'age'])]
        
        for col in outlier_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Cap outliers instead of removing them
            data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
            data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])
        
        # Convert categorical variables to appropriate types
        for col in cat_cols:
            if col not in date_columns:  # Skip date columns
                data[col] = data[col].astype('category')
        
        # Ensure all province and zipcode values are standardized (uppercase)
        if 'province' in data.columns:
            data['province'] = data['province'].str.upper()
        
        if 'zipcode' in data.columns:
            data['zipcode'] = data['zipcode'].str.upper()
        
        self.processed_data = data
        logger.info("Data cleaning completed")
        return data
    
    def feature_engineering(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Create new features from existing data based on the insurance dataset from Feb 2014 to Aug 2015.
        
        Args:
            data: DataFrame to process, if None, uses self.processed_data or self.raw_data
            
        Returns:
            DataFrame with new features
        """
        if data is None:
            if self.processed_data is not None:
                data = self.processed_data.copy()
            elif self.raw_data is not None:
                data = self.raw_data.copy()
            else:
                raise ValueError("No data loaded. Call load_data first.")
        
        logger.info("Performing feature engineering...")
        
        # Convert date columns to datetime format
        date_columns = ['transaction_date', 'policy_start_date', 'policy_end_date', 'claim_date']
        for col in date_columns:
            if col in data.columns:
                if not pd.api.types.is_datetime64_any_dtype(data[col]):
                    data[col] = pd.to_datetime(data[col], errors='coerce')
        
        # 1. Create client age groups
        if 'client_age' in data.columns:
            data['age_group'] = pd.cut(
                data['client_age'],
                bins=[0, 25, 35, 45, 55, 65, 100],
                labels=['<25', '25-35', '35-45', '45-55', '55-65', '65+'],
                right=False
            )
        
        # 2. Create vehicle age groups
        if 'car_age' in data.columns:
            data['vehicle_age_group'] = pd.cut(
                data['car_age'],
                bins=[0, 3, 6, 10, 15, 100],
                labels=['<3', '3-6', '6-10', '10-15', '15+'],
                right=False
            )
        
        # 3. Calculate policy duration in days
        if 'policy_start_date' in data.columns and 'policy_end_date' in data.columns:
            data['policy_duration_days'] = (data['policy_end_date'] - data['policy_start_date']).dt.days
        
        # 4. Calculate days between transaction and policy start
        if 'transaction_date' in data.columns and 'policy_start_date' in data.columns:
            data['days_before_policy_start'] = (data['policy_start_date'] - data['transaction_date']).dt.days
        
        # 5. Calculate days between claim and policy start
        if 'claim_date' in data.columns and 'policy_start_date' in data.columns:
            data['days_since_policy_start'] = (data['claim_date'] - data['policy_start_date']).dt.days
        
        # 6. Create claim frequency ratio (if multiple policies per client)
        if 'client_id' in data.columns and 'claim_amount' in data.columns:
            # Count claims per client
            claims_per_client = data.groupby('client_id')['claim_amount'].count()
            # Count policies per client
            policies_per_client = data.groupby('client_id')['policy_id'].nunique()
            # Create a mapping dictionary
            claim_freq_dict = (claims_per_client / policies_per_client).to_dict()
            # Map to original dataframe
            data['claim_frequency_ratio'] = data['client_id'].map(claim_freq_dict)
        
        # 7. Create premium to coverage ratio
        if 'premium_amount' in data.columns and 'coverage_amount' in data.columns:
            data['premium_to_coverage_ratio'] = data['premium_amount'] / data['coverage_amount']
        
        # 8. Create risk score based on multiple factors
        risk_factors = []
        weights = []
        
        # Add claim amount if available
        if 'claim_amount' in data.columns:
            claim_max = data['claim_amount'].max()
            if claim_max > 0:  # Avoid division by zero
                risk_factors.append(data['claim_amount'] / claim_max)
                weights.append(0.4)
        
        # Add client age if available (younger clients typically higher risk)
        if 'client_age' in data.columns:
            age_max = data['client_age'].max()
            if age_max > 0:  # Avoid division by zero
                risk_factors.append(1 - (data['client_age'] / age_max))
                weights.append(0.2)
        
        # Add car age if available
        if 'car_age' in data.columns:
            car_age_max = data['car_age'].max()
            if car_age_max > 0:  # Avoid division by zero
                risk_factors.append(data['car_age'] / car_age_max)
                weights.append(0.1)
        
        # Add car value if available (higher value cars may have higher risk)
        if 'car_value' in data.columns:
            car_value_max = data['car_value'].max()
            if car_value_max > 0:  # Avoid division by zero
                risk_factors.append(data['car_value'] / car_value_max)
                weights.append(0.15)
        
        # Add claim frequency if we calculated it
        if 'claim_frequency_ratio' in data.columns:
            freq_max = data['claim_frequency_ratio'].max()
            if freq_max > 0:  # Avoid division by zero
                risk_factors.append(data['claim_frequency_ratio'] / freq_max)
                weights.append(0.15)
        
        # Calculate weighted risk score if we have factors
        if risk_factors:
            # Normalize weights to sum to 1
            weights = [w/sum(weights) for w in weights]
            # Calculate weighted sum
            data['risk_score'] = sum(f * w for f, w in zip(risk_factors, weights)) * 100
        
        logger.info("Feature engineering completed")
        return data
    
    def split_data(self, data: Optional[pd.DataFrame] = None, 
                  target_col: str = 'claim_amount',
                  test_size: float = 0.2,
                  random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into training and testing sets.
        
        Args:
            data: DataFrame to split, if None, uses self.processed_data
            target_col: Name of the target column
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        from sklearn.model_selection import train_test_split
        
        if data is None:
            if self.processed_data is None:
                raise ValueError("No processed data available. Call clean_data first.")
            data = self.processed_data.copy()
        
        if target_col not in data.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        X = data.drop(columns=[target_col])
        y = data[target_col]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split into train ({X_train.shape[0]} rows) and test ({X_test.shape[0]} rows) sets")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, filename: str) -> None:
        """Save processed data to a CSV file.
        
        Args:
            filename: Name of the file to save the data to
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call clean_data first.")
        
        file_path = os.path.join(self.data_dir, filename)
        self.processed_data.to_csv(file_path, index=False)
        logger.info(f"Processed data saved to {file_path}")
    
    def generate_data_summary(self, data: Optional[pd.DataFrame] = None, output_file: Optional[str] = None) -> Dict:
        """Generate a comprehensive summary of the data.
        
        Args:
            data: DataFrame to summarize, if None, uses self.processed_data or self.raw_data
            output_file: If provided, save the summary to this file
            
        Returns:
            Dictionary containing summary statistics and information
        """
        if data is None:
            if self.processed_data is not None:
                data = self.processed_data.copy()
            elif self.raw_data is not None:
                data = self.raw_data.copy()
            else:
                raise ValueError("No data loaded. Call load_data first.")
        
        logger.info("Generating data summary...")
        
        summary = {}
        
        # Basic dataset information
        summary['dataset_info'] = {
            'num_rows': data.shape[0],
            'num_columns': data.shape[1],
            'memory_usage': f"{data.memory_usage(deep=True).sum() / (1024 * 1024):.2f} MB"
        }
        
        # Time range information if date columns exist
        date_columns = ['transaction_date', 'policy_start_date', 'policy_end_date', 'claim_date']
        summary['time_range'] = {}
        for col in date_columns:
            if col in data.columns and pd.api.types.is_datetime64_any_dtype(data[col]):
                summary['time_range'][col] = {
                    'min': data[col].min().strftime('%Y-%m-%d'),
                    'max': data[col].max().strftime('%Y-%m-%d')
                }
        
        # Client demographics
        summary['client_demographics'] = {}
        if 'client_id' in data.columns:
            summary['client_demographics']['unique_clients'] = data['client_id'].nunique()
        
        if 'client_age' in data.columns:
            summary['client_demographics']['age_stats'] = {
                'min': data['client_age'].min(),
                'max': data['client_age'].max(),
                'mean': data['client_age'].mean(),
                'median': data['client_age'].median()
            }
        
        if 'client_gender' in data.columns:
            summary['client_demographics']['gender_distribution'] = data['client_gender'].value_counts().to_dict()
        
        # Policy information
        summary['policy_info'] = {}
        if 'policy_id' in data.columns:
            summary['policy_info']['unique_policies'] = data['policy_id'].nunique()
        
        if 'plan_type' in data.columns:
            summary['policy_info']['plan_distribution'] = data['plan_type'].value_counts().to_dict()
        
        # Location information
        summary['location_info'] = {}
        if 'province' in data.columns:
            summary['location_info']['province_distribution'] = data['province'].value_counts().to_dict()
        
        if 'zipcode' in data.columns:
            summary['location_info']['unique_zipcodes'] = data['zipcode'].nunique()
        
        # Car information
        summary['car_info'] = {}
        if 'car_model' in data.columns:
            summary['car_info']['model_distribution'] = data['car_model'].value_counts().head(10).to_dict()
        
        if 'car_age' in data.columns:
            summary['car_info']['age_stats'] = {
                'min': data['car_age'].min(),
                'max': data['car_age'].max(),
                'mean': data['car_age'].mean(),
                'median': data['car_age'].median()
            }
        
        # Financial information
        summary['financial_info'] = {}
        for col in ['premium_amount', 'coverage_amount', 'claim_amount']:
            if col in data.columns:
                summary['financial_info'][col] = {
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'mean': data[col].mean(),
                    'median': data[col].median(),
                    'sum': data[col].sum()
                }
        
        # Claims information
        summary['claims_info'] = {}
        if 'claim_amount' in data.columns:
            # Count of claims (non-zero claim amounts)
            claims = data[data['claim_amount'] > 0]
            summary['claims_info']['total_claims'] = len(claims)
            summary['claims_info']['claim_rate'] = len(claims) / len(data)
            
            if len(claims) > 0 and 'province' in data.columns:
                summary['claims_info']['claims_by_province'] = claims['province'].value_counts().to_dict()
        
        # Save summary to file if requested
        if output_file:
            import json
            output_path = os.path.join(self.data_dir, output_file)
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=4)
            logger.info(f"Data summary saved to {output_path}")
        
        return summary

# Example usage
if __name__ == "__main__":
    processor = DataProcessor()