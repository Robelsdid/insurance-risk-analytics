"""
Exploratory Data Analysis Module for Insurance Risk Analytics

This module handles comprehensive EDA including data summarization, quality assessment,
univariate/bivariate analysis, and outlier detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class InsuranceEDAAnalyzer:
    """
    A comprehensive EDA analyzer for insurance risk analytics.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize the EDA analyzer with insurance data.
        
        Args:
            data (pd.DataFrame): Insurance data to analyze
        """
        self.data = data
        self.numerical_cols = []
        self.categorical_cols = []
        self.date_cols = []
        self._categorize_columns()
        
    def _categorize_columns(self):
        """Categorize columns by data type for analysis."""
        for col in self.data.columns:
            if self.data[col].dtype in ['int64', 'float64']:
                self.numerical_cols.append(col)
            elif self.data[col].dtype == 'object':
                if self.data[col].dtype == 'datetime64[ns]' or 'date' in col.lower():
                    self.date_cols.append(col)
                else:
                    self.categorical_cols.append(col)
    
    def get_data_summary(self) -> Dict:
        """
        Get comprehensive data summary including descriptive statistics.
        
        Returns:
            Dict containing data summary information
        """
        summary = {
            'basic_info': {
                'shape': self.data.shape,
                'total_records': len(self.data),
                'total_columns': len(self.data.columns)
            },
            'column_types': {
                'numerical': len(self.numerical_cols),
                'categorical': len(self.categorical_cols),
                'date': len(self.date_cols)
            },
            'missing_values': self.data.isnull().sum().to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum()
        }
        
        return summary
    
    def get_descriptive_statistics(self) -> pd.DataFrame:
        """
        Get descriptive statistics for numerical variables.
        
        Returns:
            pd.DataFrame: Descriptive statistics
        """
        if not self.numerical_cols:
            logger.warning("No numerical columns found for descriptive statistics")
            return pd.DataFrame()
        
        desc_stats = self.data[self.numerical_cols].describe()
        
        # Add additional statistics
        desc_stats.loc['skewness'] = self.data[self.numerical_cols].skew()
        desc_stats.loc['kurtosis'] = self.data[self.numerical_cols].kurtosis()
        desc_stats.loc['missing_count'] = self.data[self.numerical_cols].isnull().sum()
        
        return desc_stats
    
    def analyze_data_quality(self) -> Dict:
        """
        Comprehensive data quality assessment.
        
        Returns:
            Dict containing data quality metrics
        """
        quality_report = {
            'missing_data': {
                'total_missing': self.data.isnull().sum().sum(),
                'missing_percentage': (self.data.isnull().sum().sum() / (len(self.data) * len(self.data.columns))) * 100,
                'columns_with_missing': self.data.columns[self.data.isnull().any()].tolist()
            },
            'duplicate_records': {
                'total_duplicates': self.data.duplicated().sum(),
                'duplicate_percentage': (self.data.duplicated().sum() / len(self.data)) * 100
            },
            'data_types': self.data.dtypes.to_dict()
        }
        
        return quality_report
    
    def calculate_loss_ratio(self, by_group: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate Loss Ratio (TotalClaims / TotalPremium) overall and by groups.
        
        Args:
            by_group (str): Column name to group by (e.g., 'Province', 'VehicleType', 'Gender')
            
        Returns:
            pd.DataFrame: Loss ratio calculations
        """
        if 'TotalClaims' not in self.data.columns or 'TotalPremium' not in self.data.columns:
            logger.error("TotalClaims or TotalPremium columns not found")
            return pd.DataFrame()
        
        # Overall loss ratio
        overall_loss_ratio = self.data['TotalClaims'].sum() / self.data['TotalPremium'].sum()
        
        if by_group and by_group in self.data.columns:
            # Group by specified column
            grouped = self.data.groupby(by_group).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            })
            grouped['LossRatio'] = grouped['TotalClaims'] / grouped['TotalPremium']
            grouped = grouped.sort_values('LossRatio', ascending=False)
            
            return grouped
        else:
            # Return overall loss ratio
            return pd.DataFrame({
                'Metric': ['Overall Loss Ratio'],
                'Value': [overall_loss_ratio]
            })
    
    def detect_outliers(self, columns: Optional[List[str]] = None, method: str = 'iqr') -> Dict:
        """
        Detect outliers in numerical columns using specified method.
        
        Args:
            columns (List[str]): Columns to analyze for outliers
            method (str): Method to use ('iqr' or 'zscore')
            
        Returns:
            Dict containing outlier information
        """
        if columns is None:
            columns = self.numerical_cols
        
        outliers = {}
        
        for col in columns:
            if col not in self.numerical_cols:
                continue
                
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                
            elif method == 'zscore':
                z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
                outlier_mask = z_scores > 3
            else:
                logger.warning(f"Unknown method: {method}. Using IQR method.")
                continue
            
            outliers[col] = {
                'outlier_count': outlier_mask.sum(),
                'outlier_percentage': (outlier_mask.sum() / len(self.data)) * 100,
                'outlier_indices': self.data[outlier_mask].index.tolist()
            }
        
        return outliers
    
    def analyze_temporal_trends(self, date_column: str = 'TransactionMonth') -> Dict:
        """
        Analyze temporal trends in claims and premiums.
        
        Args:
            date_column (str): Column containing date information
            
        Returns:
            Dict containing temporal analysis results
        """
        if date_column not in self.data.columns:
            logger.error(f"Date column {date_column} not found")
            return {}
        
        try:
            # Convert to datetime if not already
            if self.data[date_column].dtype != 'datetime64[ns]':
                self.data[date_column] = pd.to_datetime(self.data[date_column])
            
            # Monthly aggregation
            monthly_data = self.data.groupby(self.data[date_column].dt.to_period('M')).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            })
            
            monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
            monthly_data['ClaimFrequency'] = monthly_data['TotalClaims'] / len(self.data) * 12  # Annualized
            
            return {
                'monthly_summary': monthly_data,
                'trend_analysis': {
                    'total_months': len(monthly_data),
                    'date_range': f"{monthly_data.index.min()} to {monthly_data.index.max()}"
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing temporal trends: {str(e)}")
            return {}
    
    def get_geographic_analysis(self) -> Dict:
        """
        Analyze insurance patterns by geographic location.
        
        Returns:
            Dict containing geographic analysis results
        """
        geo_analysis = {}
        
        # Province analysis
        if 'Province' in self.data.columns:
            province_stats = self.data.groupby('Province').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).rename(columns={'PolicyID': 'PolicyCount'})
            province_stats['LossRatio'] = province_stats['TotalClaims'] / province_stats['TotalPremium']
            geo_analysis['province'] = province_stats.sort_values('LossRatio', ascending=False)
        
        # Postal code analysis
        if 'PostalCode' in self.data.columns:
            postal_stats = self.data.groupby('PostalCode').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).rename(columns={'PolicyID': 'PolicyCount'})
            postal_stats['LossRatio'] = postal_stats['TotalClaims'] / postal_stats['TotalPremium']
            geo_analysis['postal_code'] = postal_stats.sort_values('LossRatio', ascending=False)
        
        return geo_analysis
    
    def get_vehicle_analysis(self) -> Dict:
        """
        Analyze insurance patterns by vehicle characteristics.
        
        Returns:
            Dict containing vehicle analysis results
        """
        vehicle_analysis = {}
        
        # Vehicle make analysis
        if 'Make' in self.data.columns:
            make_stats = self.data.groupby('Make').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).rename(columns={'PolicyID': 'PolicyCount'})
            make_stats['LossRatio'] = make_stats['TotalClaims'] / make_stats['TotalPremium']
            vehicle_analysis['make'] = make_stats.sort_values('LossRatio', ascending=False)
        
        # Vehicle type analysis
        if 'VehicleType' in self.data.columns:
            type_stats = self.data.groupby('VehicleType').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).rename(columns={'PolicyID': 'PolicyCount'})
            type_stats['LossRatio'] = type_stats['TotalClaims'] / type_stats['TotalPremium']
            vehicle_analysis['vehicle_type'] = type_stats.sort_values('LossRatio', ascending=False)
        
        return vehicle_analysis


if __name__ == "__main__":
    # Example usage
    print("EDA Analyzer module loaded successfully")
