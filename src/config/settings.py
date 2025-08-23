"""
Configuration settings for Insurance Risk Analytics 
"""

import os
from pathlib import Path

# paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
PLOTS_DIR = REPORTS_DIR / "plots"

# Data file configuration
DATA_FILE = "MachineLearningRating_v3.txt"
DATA_PATH = DATA_DIR / DATA_FILE

# Analysis parameters
DEFAULT_FIGURE_SIZE = (12, 8)
DEFAULT_COLOR_PALETTE = "husl"
DEFAULT_DPI = 300

# Statistical parameters
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3.0
CONFIDENCE_LEVEL = 0.95

# Geographic analysis
GEOGRAPHIC_COLUMNS = ['Province', 'PostalCode', 'Country', 'MainCrestaZone', 'SubCrestaZone']

# Vehicle analysis
VEHICLE_COLUMNS = ['Make', 'Model', 'VehicleType', 'Bodytype', 'RegistrationYear', 'Cylinders']

# Client analysis
CLIENT_COLUMNS = ['Gender', 'MaritalStatus', 'Age', 'Citizenship', 'LegalType']

# Financial columns
FINANCIAL_COLUMNS = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm']

# Date columns
DATE_COLUMNS = ['TransactionMonth', 'VehicleIntroDate']

# Risk analysis thresholds
HIGH_RISK_LOSS_RATIO = 0.8
MEDIUM_RISK_LOSS_RATIO = 0.5
LOW_RISK_LOSS_RATIO = 0.2

# Plot themes
PLOT_THEMES = {
    'default': {
        'style': 'seaborn-v0_8',
        'palette': 'husl',
        'font_size': 12,
        'title_font_size': 16
    },
    'professional': {
        'style': 'classic',
        'palette': 'Set2',
        'font_size': 10,
        'title_font_size': 14
    }
}

# Logging configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = REPORTS_DIR / "analysis.log"

# Ensure directories exist
def ensure_directories():
    """Ensure all required directories exist."""
    for directory in [DATA_DIR, REPORTS_DIR, PLOTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

# Initialize directories
ensure_directories()
