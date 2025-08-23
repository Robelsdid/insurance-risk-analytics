"""
Plot Generator Module for Insurance Risk Analytics

This module creates creative and beautiful visualizations for EDA insights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)


class InsurancePlotGenerator:
    """
    A class to generate creative and beautiful plots for insurance risk analytics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the plot generator.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for plots
        """
        self.figsize = figsize
        self.colors = sns.color_palette("husl", 10)
        
    def create_loss_ratio_heatmap(self, data: pd.DataFrame, 
                                 group1: str = 'Province', 
                                 group2: str = 'VehicleType') -> plt.Figure:
        """
        Create a heatmap showing loss ratios across different groups.
        
        Args:
            data (pd.DataFrame): Insurance data
            group1 (str): First grouping variable
            group2 (str): Second grouping variable
            
        Returns:
            plt.Figure: Generated heatmap figure
        """
        try:
            # Calculate loss ratios for the two groups
            pivot_data = data.groupby([group1, group2]).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum'
            }).reset_index()
            
            pivot_data['LossRatio'] = pivot_data['TotalClaims'] / pivot_data['TotalPremium']
            
            # Create pivot table for heatmap
            heatmap_data = pivot_data.pivot(index=group1, columns=group2, values='LossRatio')
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create heatmap
            sns.heatmap(heatmap_data, 
                       annot=True, 
                       fmt='.3f', 
                       cmap='RdYlBu_r', 
                       center=heatmap_data.mean().mean(),
                       cbar_kws={'label': 'Loss Ratio'},
                       ax=ax)
            
            ax.set_title(f'Loss Ratio Heatmap: {group1} vs {group2}', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel(group2, fontsize=12, fontweight='bold')
            ax.set_ylabel(group1, fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating loss ratio heatmap: {str(e)}")
            return None
    
    def create_premium_claims_scatter(self, data: pd.DataFrame, 
                                    color_by: str = 'Province',
                                    size_by: str = 'CustomValueEstimate') -> plt.Figure:
        """
        Create a scatter plot showing relationship between premiums and claims.
        
        Args:
            data (pd.DataFrame): Insurance data
            color_by (str): Column to color points by
            size_by (str): Column to size points by
            
        Returns:
            plt.Figure: Generated scatter plot figure
        """
        try:
            # Filter out rows with missing values
            plot_data = data[[color_by, size_by, 'TotalPremium', 'TotalClaims']].dropna()
            
            # Create the plot
            fig, ax = plt.subplots(figsize=self.figsize)
            
            # Create scatter plot
            scatter = ax.scatter(plot_data['TotalPremium'], 
                               plot_data['TotalClaims'],
                               c=plot_data[color_by].astype('category').cat.codes,
                               s=plot_data[size_by] / plot_data[size_by].max() * 200 + 20,
                               alpha=0.6,
                               cmap='viridis')
            
            # Add trend line
            z = np.polyfit(plot_data['TotalPremium'], plot_data['TotalClaims'], 1)
            p = np.poly1d(z)
            ax.plot(plot_data['TotalPremium'], p(plot_data['TotalPremium']), 
                   "r--", alpha=0.8, linewidth=2)
            
            # Add diagonal line (1:1 ratio)
            max_val = max(plot_data['TotalPremium'].max(), plot_data['TotalClaims'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, linewidth=1)
            
            # Customize the plot
            ax.set_xlabel('Total Premium (ZAR)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Total Claims (ZAR)', fontsize=12, fontweight='bold')
            ax.set_title('Premium vs Claims Relationship', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_by, fontsize=10)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating premium-claims scatter plot: {str(e)}")
            return None
    
    def create_temporal_trend_plot(self, data: pd.DataFrame, 
                                  date_column: str = 'TransactionMonth') -> plt.Figure:
        """
        Create a temporal trend plot showing how key metrics change over time.
        
        Args:
            data (pd.DataFrame): Insurance data
            date_column (str): Column containing date information
            
        Returns:
            plt.Figure: Generated temporal trend figure
        """
        try:
            # Convert to datetime if not already
            if data[date_column].dtype != 'datetime64[ns]':
                data[date_column] = pd.to_datetime(data[date_column])
            
            # Monthly aggregation
            monthly_data = data.groupby(data[date_column].dt.to_period('M')).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            
            monthly_data['LossRatio'] = monthly_data['TotalClaims'] / monthly_data['TotalPremium']
            monthly_data['date'] = monthly_data[date_column].dt.to_timestamp()
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(self.figsize[0], self.figsize[1] * 1.5))
            
            # Top plot: Premium and Claims over time
            ax1.plot(monthly_data['date'], monthly_data['TotalPremium'], 
                    marker='o', linewidth=2, markersize=6, label='Total Premium', color='blue')
            ax1.plot(monthly_data['date'], monthly_data['TotalClaims'], 
                    marker='s', linewidth=2, markersize=6, label='Total Claims', color='red')
            
            ax1.set_title('Monthly Premium and Claims Trends', 
                         fontsize=14, fontweight='bold')
            ax1.set_ylabel('Amount (ZAR)', fontsize=12)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Loss Ratio over time
            ax2.plot(monthly_data['date'], monthly_data['LossRatio'], 
                    marker='^', linewidth=2, markersize=6, color='green')
            ax2.axhline(y=monthly_data['LossRatio'].mean(), color='red', 
                       linestyle='--', alpha=0.7, label=f'Mean: {monthly_data["LossRatio"].mean():.3f}')
            
            ax2.set_title('Monthly Loss Ratio Trend', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Date', fontsize=12)
            ax2.set_ylabel('Loss Ratio', fontsize=12)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            for ax in [ax1, ax2]:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating temporal trend plot: {str(e)}")
            return None
    
    def create_geographic_risk_map(self, data: pd.DataFrame, 
                                  location_col: str = 'Province') -> plt.Figure:
        """
        Create a geographic risk visualization showing loss ratios by location.
        
        Args:
            data (pd.DataFrame): Insurance data
            location_col (str): Column containing location information
            
        Returns:
            plt.Figure: Generated geographic risk figure
        """
        try:
            # Calculate loss ratios by location
            location_stats = data.groupby(location_col).agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            
            location_stats['LossRatio'] = location_stats['TotalClaims'] / location_stats['TotalPremium']
            location_stats['PolicyCount'] = location_stats['PolicyID']
            
            # Sort by loss ratio
            location_stats = location_stats.sort_values('LossRatio', ascending=False)
            
            # Create the plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1]))
            
            # Left plot: Loss Ratio by Location
            bars1 = ax1.bar(range(len(location_stats)), location_stats['LossRatio'], 
                           color=self.colors[:len(location_stats)])
            ax1.set_title('Loss Ratio by Location', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Loss Ratio', fontsize=12)
            ax1.set_xticks(range(len(location_stats)))
            ax1.set_xticklabels(location_stats[location_col], rotation=45, ha='right')
            
            # Add value labels on bars
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Right plot: Policy Count vs Loss Ratio
            scatter = ax2.scatter(location_stats['PolicyCount'], location_stats['LossRatio'],
                                s=100, c=range(len(location_stats)), cmap='viridis', alpha=0.7)
            
            # Add location labels
            for i, row in location_stats.iterrows():
                ax2.annotate(row[location_col], 
                           (row['PolicyCount'], row['LossRatio']),
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax2.set_title('Policy Count vs Loss Ratio', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Number of Policies', fontsize=12)
            ax2.set_ylabel('Loss Ratio', fontsize=12)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating geographic risk map: {str(e)}")
            return None
    
    def create_vehicle_risk_analysis(self, data: pd.DataFrame) -> plt.Figure:
        """
        Create a comprehensive vehicle risk analysis visualization.
        
        Args:
            data (pd.DataFrame): Insurance data
            
        Returns:
            plt.Figure: Generated vehicle risk analysis figure
        """
        try:
            # Analyze vehicle make and type
            make_stats = data.groupby('Make').agg({
                'TotalClaims': 'sum',
                'TotalPremium': 'sum',
                'PolicyID': 'count'
            }).reset_index()
            make_stats['LossRatio'] = make_stats['TotalClaims'] / make_stats['TotalPremium']
            make_stats = make_stats.sort_values('LossRatio', ascending=False).head(10)
            
            # Create the plot
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(self.figsize[0] * 1.5, self.figsize[1] * 1.5))
            
            # Top left: Top 10 Vehicle Makes by Loss Ratio
            bars1 = ax1.barh(range(len(make_stats)), make_stats['LossRatio'], 
                            color=self.colors[:len(make_stats)])
            ax1.set_title('Top 10 Vehicle Makes by Loss Ratio', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Loss Ratio', fontsize=10)
            ax1.set_yticks(range(len(make_stats)))
            ax1.set_yticklabels(make_stats['Make'])
            
            # Top right: Premium vs Claims for Top Makes
            ax2.scatter(make_stats['TotalPremium'], make_stats['TotalClaims'], 
                       s=make_stats['PolicyID'] * 2, alpha=0.7, c=range(len(make_stats)), cmap='viridis')
            for i, row in make_stats.iterrows():
                ax2.annotate(row['Make'], (row['TotalPremium'], row['TotalClaims']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax2.set_title('Premium vs Claims by Vehicle Make', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Total Premium', fontsize=10)
            ax2.set_ylabel('Total Claims', fontsize=10)
            
            # Bottom left: Policy Count Distribution
            ax3.hist(data['Make'].value_counts().values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.set_title('Distribution of Policies by Vehicle Make', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Number of Policies', fontsize=10)
            ax3.set_ylabel('Frequency', fontsize=10)
            
            # Bottom right: Loss Ratio vs Policy Count
            ax4.scatter(make_stats['PolicyID'], make_stats['LossRatio'], 
                       s=100, alpha=0.7, c=range(len(make_stats)), cmap='viridis')
            for i, row in make_stats.iterrows():
                ax4.annotate(row['Make'], (row['PolicyID'], row['LossRatio']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            ax4.set_title('Policy Count vs Loss Ratio', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Number of Policies', fontsize=10)
            ax4.set_ylabel('Loss Ratio', fontsize=10)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error creating vehicle risk analysis: {str(e)}")
            return None
    
    def save_all_plots(self, data: pd.DataFrame, save_dir: str = "reports/plots") -> List[str]:
        """
        Generate and save all required plots for the EDA task.
        
        Args:
            data (pd.DataFrame): Insurance data
            save_dir (str): Directory to save plots
            
        Returns:
            List[str]: List of saved plot file paths
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        saved_files = []
        
        # Generate the three required creative plots
        plots = [
            ('loss_ratio_heatmap', self.create_loss_ratio_heatmap(data)),
            ('premium_claims_scatter', self.create_premium_claims_scatter(data)),
            ('temporal_trends', self.create_temporal_trend_plot(data))
        ]
        
        for plot_name, fig in plots:
            if fig is not None:
                file_path = os.path.join(save_dir, f"{plot_name}.png")
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(file_path)
                logger.info(f"Saved plot: {file_path}")
        
        return saved_files


if __name__ == "__main__":
    print("Plot Generator module loaded successfully")
