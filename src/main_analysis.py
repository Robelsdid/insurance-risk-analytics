"""
Main Analysis Runner for Insurance Risk Analytics

This module orchestrates the complete EDA process using all modular components.
"""

import sys
import os
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from data.data_loader import InsuranceDataLoader
from analysis.eda_analyzer import InsuranceEDAAnalyzer
from visualization.plot_generator import InsurancePlotGenerator
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_complete_eda():
    """
    Run the complete EDA analysis for the insurance risk analytics project.
    """
    try:
        logger.info("Starting Insurance Risk Analytics EDA")
        
        # Step 1: Load the data
        logger.info("Step 1: Loading insurance data...")
        loader = InsuranceDataLoader()
        data = loader.load_data()
        
        # Display basic data information
        data_info = loader.get_data_info()
        logger.info(f"Data loaded successfully: {data_info['shape'][0]} records, {data_info['shape'][1]} columns")
        
        # Step 2: Initialize EDA Analyzer
        logger.info("Step 2: Initializing EDA Analyzer...")
        eda_analyzer = InsuranceEDAAnalyzer(data)
        
        # Step 3: Perform comprehensive EDA
        logger.info("Step 3: Performing comprehensive EDA...")
        
        # Get data summary
        summary = eda_analyzer.get_data_summary()
        logger.info(f"Data Summary: {summary['basic_info']}")
        
        # Get descriptive statistics
        desc_stats = eda_analyzer.get_descriptive_statistics()
        logger.info("Descriptive statistics calculated")
        
        # Analyze data quality
        quality_report = eda_analyzer.analyze_data_quality()
        logger.info(f"Data Quality: {quality_report['missing_data']['missing_percentage']:.2f}% missing data")
        
        # Calculate loss ratios
        overall_loss_ratio = eda_analyzer.calculate_loss_ratio()
        logger.info(f"Overall Loss Ratio: {overall_loss_ratio.iloc[0]['Value']:.4f}")
        
        # Geographic analysis
        geo_analysis = eda_analyzer.get_geographic_analysis()
        if 'province' in geo_analysis:
            logger.info(f"Province analysis completed for {len(geo_analysis['province'])} provinces")
        
        # Vehicle analysis
        vehicle_analysis = eda_analyzer.get_vehicle_analysis()
        if 'make' in vehicle_analysis:
            logger.info(f"Vehicle make analysis completed for {len(vehicle_analysis['make'])} makes")
        
        # Temporal analysis
        temporal_analysis = eda_analyzer.analyze_temporal_trends()
        if temporal_analysis:
            logger.info("Temporal trends analysis completed")
        
        # Outlier detection
        outliers = eda_analyzer.detect_outliers()
        logger.info(f"Outlier detection completed for {len(outliers)} columns")
        
        # Step 4: Generate visualizations
        logger.info("Step 4: Generating visualizations...")
        plot_generator = InsurancePlotGenerator()
        
        # Create and save the three required creative plots
        saved_plots = plot_generator.save_all_plots(data)
        logger.info(f"Generated and saved {len(saved_plots)} plots: {saved_plots}")
        
        # Step 5: Generate comprehensive report
        logger.info("Step 5: Generating comprehensive EDA report...")
        generate_eda_report(summary, quality_report, desc_stats, geo_analysis, 
                          vehicle_analysis, temporal_analysis, outliers)
        
        logger.info("EDA analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error during EDA analysis: {str(e)}")
        return False


def generate_eda_report(summary, quality_report, desc_stats, geo_analysis, 
                       vehicle_analysis, temporal_analysis, outliers):
    """
    Generate a comprehensive EDA report.
    
    Args:
        summary: Data summary information
        quality_report: Data quality assessment
        desc_stats: Descriptive statistics
        geo_analysis: Geographic analysis results
        vehicle_analysis: Vehicle analysis results
        temporal_analysis: Temporal analysis results
        outliers: Outlier detection results
    """
    try:
        # Create reports directory
        os.makedirs("reports", exist_ok=True)
        
        report_path = "reports/eda_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("INSURANCE RISK ANALYTICS - EXPLORATORY DATA ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Data Overview
            f.write("1. DATA OVERVIEW\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Records: {summary['basic_info']['total_records']:,}\n")
            f.write(f"Total Columns: {summary['basic_info']['total_columns']}\n")
            f.write(f"Numerical Columns: {summary['column_types']['numerical']}\n")
            f.write(f"Categorical Columns: {summary['column_types']['categorical']}\n")
            f.write(f"Date Columns: {summary['column_types']['date']}\n")
            f.write(f"Memory Usage: {summary['memory_usage'] / 1024 / 1024:.2f} MB\n\n")
            
            # Data Quality
            f.write("2. DATA QUALITY ASSESSMENT\n")
            f.write("-" * 40 + "\n")
            f.write(f"Missing Data: {quality_report['missing_data']['total_missing']:,} values\n")
            f.write(f"Missing Percentage: {quality_report['missing_data']['missing_percentage']:.2f}%\n")
            f.write(f"Duplicate Records: {quality_report['duplicate_records']['total_duplicates']:,}\n")
            f.write(f"Duplicate Percentage: {quality_report['duplicate_records']['duplicate_percentage']:.2f}%\n\n")
            
            # Key Insights
            f.write("3. KEY INSIGHTS\n")
            f.write("-" * 40 + "\n")
            
            # Geographic insights
            if 'province' in geo_analysis:
                f.write("Geographic Risk Analysis:\n")
                top_risk_province = geo_analysis['province'].iloc[0]
                f.write(f"  - Highest Risk Province: {top_risk_province.name} (Loss Ratio: {top_risk_province['LossRatio']:.4f})\n")
                f.write(f"  - Policies in {top_risk_province.name}: {top_risk_province['PolicyCount']:,}\n\n")
            
            # Vehicle insights
            if 'make' in vehicle_analysis:
                f.write("Vehicle Risk Analysis:\n")
                top_risk_make = vehicle_analysis['make'].iloc[0]
                f.write(f"  - Highest Risk Vehicle Make: {top_risk_make.name} (Loss Ratio: {top_risk_make['LossRatio']:.4f})\n")
                f.write(f"  - Policies for {top_risk_make.name}: {top_risk_make['PolicyCount']:,}\n\n")
            
            # Temporal insights
            if temporal_analysis:
                f.write("Temporal Trends:\n")
                f.write(f"  - Analysis Period: {temporal_analysis['trend_analysis']['date_range']}\n")
                f.write(f"  - Total Months Analyzed: {temporal_analysis['trend_analysis']['total_months']}\n\n")
            
            # Outlier insights
            if outliers:
                f.write("Outlier Detection:\n")
                for col, outlier_info in outliers.items():
                    if outlier_info['outlier_count'] > 0:
                        f.write(f"  - {col}: {outlier_info['outlier_count']:,} outliers ({outlier_info['outlier_percentage']:.2f}%)\n")
                f.write("\n")
            
            # Recommendations
            f.write("4. RECOMMENDATIONS\n")
            f.write("-" * 40 + "\n")
            f.write("Based on the EDA analysis:\n")
            f.write("1. Focus on high-risk geographic areas for targeted risk management\n")
            f.write("2. Investigate vehicle makes with high loss ratios for premium adjustments\n")
            f.write("3. Monitor temporal trends for seasonal risk patterns\n")
            f.write("4. Address data quality issues to improve analysis accuracy\n")
            f.write("5. Consider segmenting policies based on risk factors identified\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Report generated successfully!\n")
            f.write("=" * 80 + "\n")
        
        logger.info(f"EDA report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Error generating EDA report: {str(e)}")


if __name__ == "__main__":
    # Run the complete EDA analysis
    success = run_complete_eda()
    
    if success:
        print("\n" + "="*60)
        print("EDA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Check the following outputs:")
        print("- reports/eda_report.txt (Comprehensive analysis report)")
        print("- reports/plots/ (Generated visualizations)")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("EDA ANALYSIS FAILED!")
        print("Check the logs above for error details.")
        print("="*60)
