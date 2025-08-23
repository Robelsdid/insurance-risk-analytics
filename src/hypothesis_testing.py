import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import logging
from typing import Dict, List, Tuple, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HypothesisTester:
    """Class for performing A/B hypothesis testing on insurance data."""
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """Initialize the HypothesisTester.
        
        Args:
            data: DataFrame containing the insurance data
        """
        self.data = data
        self.test_results = {}
        
    def load_data(self, data: pd.DataFrame) -> None:
        """Load data into the tester.
        
        Args:
            data: DataFrame containing the insurance data
        """
        self.data = data
        logger.info(f"Data loaded with shape {data.shape}")
    
    def test_risk_by_province(self, risk_column: str = 'claim_amount', 
                             alpha: float = 0.05) -> Dict:
        """Test if there are risk differences across provinces.
        
        Null Hypothesis: There are no risk differences across provinces.
        
        Args:
            risk_column: Column name representing risk (e.g., claim_amount)
            alpha: Significance level
            
        Returns:
            Dictionary containing test results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if 'province' not in self.data.columns:
            raise ValueError("'province' column not found in data")
        
        if risk_column not in self.data.columns:
            raise ValueError(f"'{risk_column}' column not found in data")
        
        logger.info("Testing risk differences across provinces...")
        
        # Create groups by province
        provinces = self.data['province'].unique()
        groups = [self.data[self.data['province'] == province][risk_column].values 
                 for province in provinces]
        
        # Perform ANOVA test
        f_stat, p_value = stats.f_oneway(*groups)
        
        # Create a more detailed analysis using statsmodels
        formula = f"{risk_column} ~ C(province)"
        model = ols(formula, data=self.data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Calculate mean risk by province for reporting
        province_means = self.data.groupby('province')[risk_column].mean().sort_values(ascending=False)
        
        # Store results
        result = {
            'test_name': 'Risk Differences Across Provinces',
            'null_hypothesis': 'There are no risk differences across provinces',
            'test_type': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'conclusion': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis",
            'province_means': province_means.to_dict(),
            'anova_table': anova_table.to_dict(),
            'groups': {province: group.tolist() for province, group in zip(provinces, groups)}
        }
        
        self.test_results['risk_by_province'] = result
        
        logger.info(f"Province risk test completed: p-value = {p_value:.4f}, "
                   f"{'reject' if p_value < alpha else 'fail to reject'} null hypothesis")
        
        return result
    
    def test_risk_by_zipcode(self, risk_column: str = 'claim_amount', 
                            alpha: float = 0.05) -> Dict:
        """Test if there are risk differences between zipcodes.
        
        Null Hypothesis: There are no risk differences between zipcodes.
        
        Args:
            risk_column: Column name representing risk (e.g., claim_amount)
            alpha: Significance level
            
        Returns:
            Dictionary containing test results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if 'zipcode' not in self.data.columns:
            raise ValueError("'zipcode' column not found in data")
        
        if risk_column not in self.data.columns:
            raise ValueError(f"'{risk_column}' column not found in data")
        
        logger.info("Testing risk differences between zipcodes...")
        
        # For zipcodes, we might have too many groups for a simple ANOVA
        # Let's use a more sophisticated approach with statsmodels
        formula = f"{risk_column} ~ C(zipcode)"
        model = ols(formula, data=self.data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extract key statistics
        f_stat = anova_table.loc['C(zipcode)', 'F']
        p_value = anova_table.loc['C(zipcode)', 'PR(>F)']
        
        # Calculate mean risk by zipcode for reporting
        zipcode_means = self.data.groupby('zipcode')[risk_column].mean()
        
        # Identify top 10 highest and lowest risk zipcodes
        top_10_high_risk = zipcode_means.nlargest(10)
        top_10_low_risk = zipcode_means.nsmallest(10)
        
        # Store results
        result = {
            'test_name': 'Risk Differences Between Zipcodes',
            'null_hypothesis': 'There are no risk differences between zipcodes',
            'test_type': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'conclusion': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis",
            'top_10_high_risk': top_10_high_risk.to_dict(),
            'top_10_low_risk': top_10_low_risk.to_dict(),
            'anova_table': anova_table.to_dict()
        }
        
        self.test_results['risk_by_zipcode'] = result
        
        logger.info(f"Zipcode risk test completed: p-value = {p_value:.4f}, "
                   f"{'reject' if p_value < alpha else 'fail to reject'} null hypothesis")
        
        return result
    
    def test_margin_by_zipcode(self, premium_column: str = 'premium', 
                              claim_column: str = 'claim_amount',
                              alpha: float = 0.05) -> Dict:
        """Test if there are significant margin differences between zip codes.
        
        Null Hypothesis: There are no significant margin differences between zip codes.
        
        Args:
            premium_column: Column name representing premium
            claim_column: Column name representing claim amount
            alpha: Significance level
            
        Returns:
            Dictionary containing test results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if 'zipcode' not in self.data.columns:
            raise ValueError("'zipcode' column not found in data")
        
        if premium_column not in self.data.columns or claim_column not in self.data.columns:
            raise ValueError(f"'{premium_column}' or '{claim_column}' column not found in data")
        
        logger.info("Testing margin differences between zipcodes...")
        
        # Calculate margin for each policy
        self.data['margin'] = self.data[premium_column] - self.data[claim_column]
        
        # Use ANOVA to test for differences in margins between zipcodes
        formula = "margin ~ C(zipcode)"
        model = ols(formula, data=self.data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        
        # Extract key statistics
        f_stat = anova_table.loc['C(zipcode)', 'F']
        p_value = anova_table.loc['C(zipcode)', 'PR(>F)']
        
        # Calculate mean margin by zipcode for reporting
        zipcode_margins = self.data.groupby('zipcode')['margin'].mean()
        
        # Identify top 10 highest and lowest margin zipcodes
        top_10_high_margin = zipcode_margins.nlargest(10)
        top_10_low_margin = zipcode_margins.nsmallest(10)
        
        # Store results
        result = {
            'test_name': 'Margin Differences Between Zipcodes',
            'null_hypothesis': 'There are no significant margin differences between zip codes',
            'test_type': 'ANOVA',
            'f_statistic': f_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'conclusion': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis",
            'top_10_high_margin': top_10_high_margin.to_dict(),
            'top_10_low_margin': top_10_low_margin.to_dict(),
            'anova_table': anova_table.to_dict()
        }
        
        self.test_results['margin_by_zipcode'] = result
        
        logger.info(f"Zipcode margin test completed: p-value = {p_value:.4f}, "
                   f"{'reject' if p_value < alpha else 'fail to reject'} null hypothesis")
        
        return result
    
    def test_risk_by_gender(self, risk_column: str = 'claim_amount', 
                           gender_column: str = 'gender',
                           alpha: float = 0.05) -> Dict:
        """Test if there are significant risk differences between genders.
        
        Null Hypothesis: There are not significant risk differences between women and men.
        
        Args:
            risk_column: Column name representing risk (e.g., claim_amount)
            gender_column: Column name representing gender
            alpha: Significance level
            
        Returns:
            Dictionary containing test results
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data first.")
        
        if gender_column not in self.data.columns:
            raise ValueError(f"'{gender_column}' column not found in data")
        
        if risk_column not in self.data.columns:
            raise ValueError(f"'{risk_column}' column not found in data")
        
        logger.info("Testing risk differences between genders...")
        
        # Filter data to include only binary gender categories for this test
        # This is a simplification and should be adapted based on the actual data structure
        gender_data = self.data[self.data[gender_column].isin(['M', 'F'])].copy()
        
        # Create groups by gender
        male_data = gender_data[gender_data[gender_column] == 'M'][risk_column].values
        female_data = gender_data[gender_data[gender_column] == 'F'][risk_column].values
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(male_data, female_data, equal_var=False)
        
        # Calculate mean risk by gender for reporting
        gender_means = gender_data.groupby(gender_column)[risk_column].mean()
        
        # Store results
        result = {
            'test_name': 'Risk Differences Between Genders',
            'null_hypothesis': 'There are not significant risk differences between women and men',
            'test_type': 't-test',
            't_statistic': t_stat,
            'p_value': p_value,
            'alpha': alpha,
            'reject_null': p_value < alpha,
            'conclusion': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis",
            'gender_means': gender_means.to_dict(),
            'male_mean': male_data.mean(),
            'female_mean': female_data.mean(),
            'male_std': male_data.std(),
            'female_std': female_data.std(),
            'male_count': len(male_data),
            'female_count': len(female_data)
        }
        
        self.test_results['risk_by_gender'] = result
        
        logger.info(f"Gender risk test completed: p-value = {p_value:.4f}, "
                   f"{'reject' if p_value < alpha else 'fail to reject'} null hypothesis")
        
        return result
    
    def visualize_test_results(self, test_key: str, save_path: Optional[str] = None) -> None:
        """Visualize the results of a specific hypothesis test.
        
        Args:
            test_key: Key of the test to visualize
            save_path: Path to save the visualization, if None, the plot is displayed
        """
        if test_key not in self.test_results:
            raise ValueError(f"Test '{test_key}' not found in results")
        
        result = self.test_results[test_key]
        
        plt.figure(figsize=(12, 8))
        
        if test_key == 'risk_by_province':
            # Bar plot of mean risk by province
            provinces = list(result['province_means'].keys())
            means = list(result['province_means'].values())
            
            sns.barplot(x=provinces, y=means)
            plt.title(f"Mean Risk by Province (p-value: {result['p_value']:.4f})")
            plt.xlabel('Province')
            plt.ylabel('Mean Risk')
            plt.xticks(rotation=45)
            
        elif test_key == 'risk_by_zipcode':
            # Bar plot of top 10 highest and lowest risk zipcodes
            high_risk = result['top_10_high_risk']
            low_risk = result['top_10_low_risk']
            
            plt.subplot(1, 2, 1)
            sns.barplot(x=list(high_risk.keys()), y=list(high_risk.values()))
            plt.title("Top 10 Highest Risk Zipcodes")
            plt.xlabel('Zipcode')
            plt.ylabel('Mean Risk')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            sns.barplot(x=list(low_risk.keys()), y=list(low_risk.values()))
            plt.title("Top 10 Lowest Risk Zipcodes")
            plt.xlabel('Zipcode')
            plt.ylabel('Mean Risk')
            plt.xticks(rotation=45)
            
            plt.suptitle(f"Risk by Zipcode (p-value: {result['p_value']:.4f})")
            
        elif test_key == 'margin_by_zipcode':
            # Bar plot of top 10 highest and lowest margin zipcodes
            high_margin = result['top_10_high_margin']
            low_margin = result['top_10_low_margin']
            
            plt.subplot(1, 2, 1)
            sns.barplot(x=list(high_margin.keys()), y=list(high_margin.values()))
            plt.title("Top 10 Highest Margin Zipcodes")
            plt.xlabel('Zipcode')
            plt.ylabel('Mean Margin')
            plt.xticks(rotation=45)
            
            plt.subplot(1, 2, 2)
            sns.barplot(x=list(low_margin.keys()), y=list(low_margin.values()))
            plt.title("Top 10 Lowest Margin Zipcodes")
            plt.xlabel('Zipcode')
            plt.ylabel('Mean Margin')
            plt.xticks(rotation=45)
            
            plt.suptitle(f"Margin by Zipcode (p-value: {result['p_value']:.4f})")
            
        elif test_key == 'risk_by_gender':
            # Bar plot of mean risk by gender
            genders = list(result['gender_means'].keys())
            means = list(result['gender_means'].values())
            
            sns.barplot(x=genders, y=means)
            plt.title(f"Mean Risk by Gender (p-value: {result['p_value']:.4f})")
            plt.xlabel('Gender')
            plt.ylabel('Mean Risk')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Visualization saved to {save_path}")
        else:
            plt.show()
    
    def generate_report(self, output_path: str) -> None:
        """Generate a comprehensive report of all hypothesis tests.
        
        Args:
            output_path: Path to save the report
        """
        if not self.test_results:
            raise ValueError("No test results available. Run tests first.")
        
        with open(output_path, 'w') as f:
            f.write("# Hypothesis Testing Report\n\n")
            
            for test_key, result in self.test_results.items():
                f.write(f"## {result['test_name']}\n\n")
                f.write(f"**Null Hypothesis:** {result['null_hypothesis']}\n\n")
                f.write(f"**Test Type:** {result['test_type']}\n\n")
                
                if 'f_statistic' in result:
                    f.write(f"**F-statistic:** {result['f_statistic']:.4f}\n\n")
                if 't_statistic' in result:
                    f.write(f"**t-statistic:** {result['t_statistic']:.4f}\n\n")
                    
                f.write(f"**p-value:** {result['p_value']:.4f}\n\n")
                f.write(f"**Alpha:** {result['alpha']}\n\n")
                f.write(f"**Conclusion:** {result['conclusion']}\n\n")
                
                # Add specific details based on test type
                if test_key == 'risk_by_province':
                    f.write("### Mean Risk by Province\n\n")
                    for province, mean in result['province_means'].items():
                        f.write(f"- {province}: {mean:.2f}\n")
                    
                elif test_key == 'risk_by_zipcode':
                    f.write("### Top 10 Highest Risk Zipcodes\n\n")
                    for zipcode, mean in result['top_10_high_risk'].items():
                        f.write(f"- {zipcode}: {mean:.2f}\n")
                    
                    f.write("\n### Top 10 Lowest Risk Zipcodes\n\n")
                    for zipcode, mean in result['top_10_low_risk'].items():
                        f.write(f"- {zipcode}: {mean:.2f}\n")
                    
                elif test_key == 'margin_by_zipcode':
                    f.write("### Top 10 Highest Margin Zipcodes\n\n")
                    for zipcode, margin in result['top_10_high_margin'].items():
                        f.write(f"- {zipcode}: {margin:.2f}\n")
                    
                    f.write("\n### Top 10 Lowest Margin Zipcodes\n\n")
                    for zipcode, margin in result['top_10_low_margin'].items():
                        f.write(f"- {zipcode}: {margin:.2f}\n")
                    
                elif test_key == 'risk_by_gender':
                    f.write("### Mean Risk by Gender\n\n")
                    for gender, mean in result['gender_means'].items():
                        f.write(f"- {gender}: {mean:.2f}\n")
                    
                    f.write(f"\n**Male Mean:** {result['male_mean']:.2f}\n")
                    f.write(f"**Female Mean:** {result['female_mean']:.2f}\n")
                    f.write(f"**Male Std Dev:** {result['male_std']:.2f}\n")
                    f.write(f"**Female Std Dev:** {result['female_std']:.2f}\n")
                    f.write(f"**Male Count:** {result['male_count']}\n")
                    f.write(f"**Female Count:** {result['female_count']}\n")
                
                f.write("\n---\n\n")
        
        logger.info(f"Report generated and saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # This is a placeholder for demonstration
    # In a real scenario, you would load actual data
    
    # Create a sample dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Sample data with some built-in differences
    provinces = np.random.choice(['Western Cape', 'Eastern Cape', 'Gauteng', 'KwaZulu-Natal'], n_samples)
    zipcodes = np.random.choice(['1000', '2000', '3000', '4000', '5000'], n_samples)
    genders = np.random.choice(['M', 'F'], n_samples)
    
    # Create claim amounts with built-in differences
    claim_base = 5000
    province_effect = {'Western Cape': 1000, 'Eastern Cape': -500, 'Gauteng': 2000, 'KwaZulu-Natal': -1000}
    zipcode_effect = {'1000': 2000, '2000': 1000, '3000': 0, '4000': -1000, '5000': -2000}
    gender_effect = {'M': 500, 'F': -500}
    
    claims = np.array([claim_base + 
                      province_effect[provinces[i]] + 
                      zipcode_effect[zipcodes[i]] + 
                      gender_effect[genders[i]] + 
                      np.random.normal(0, 1000) 
                      for i in range(n_samples)])
    
    # Create premiums with some relationship to claims but also independent variation
    premiums = claims * 1.2 + np.random.normal(0, 500, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'province': provinces,
        'zipcode': zipcodes,
        'gender': genders,
        'claim_amount': claims,
        'premium': premiums
    })
    
    # Initialize tester and run tests
    tester = HypothesisTester(data)
    tester.test_risk_by_province()
    tester.test_risk_by_zipcode()
    tester.test_margin_by_zipcode()
    tester.test_risk_by_gender()
    
    # Generate visualizations
    # tester.visualize_test_results('risk_by_province')
    # tester.visualize_test_results('risk_by_zipcode')
    # tester.visualize_test_results('margin_by_zipcode')
    # tester.visualize_test_results('risk_by_gender')
    
    # Generate report
    # tester.generate_report('../reports/hypothesis_testing_report.md')