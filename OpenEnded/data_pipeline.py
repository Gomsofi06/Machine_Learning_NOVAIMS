# Dataframes
import pandas as pd

# Others
import sys
# setting path
sys.path.append('../')
from utils import *

def pipeline(df):
    '''Check duplicate IDs'''
    # Check datatypes
    datatype_changes([df])

    # Replace incoherencies by na
    limit_feature([df], 'Age at Injury', minimum=14, maximum=119)
    limit_feature([df], 'Birth Year', minimum=2024-119+1, maximum=2024-14+1)
    limit_feature([df], 'Average Weekly Wage', minimum=1, maximum=None)

    # Input na

    # Filter the rows where 'Accident Date' is NaN, but 'Assembly Date' is not NaN
    condition = df['Accident Date'].isna() & df['Assembly Date'].notna()
    # Replace missing 'Accident Date' with 'Assembly Date' where the condition is true
    df.loc[condition, 'Accident Date'] = df.loc[condition, 'Assembly Date']

    # Filter the rows where 'Age at Injury' is NaN, but 'Birth Year' and 'Accident Date' are not NaN
    condition = df['Age at Injury'].isna() & df['Birth Year'].notna() & df['Accident Date'].notna()
    # Replace missing 'Age at Injury' with the difference between 'Accident Date' and 'Birth Year'
    df.loc[condition, 'Age at Injury'] = df.loc[condition, 'Accident Date'].dt.year - df.loc[condition, 'Birth Year']

    # Filter the rows where 'Birth Year' is NaN, but 'Age at Injury' and 'Accident Date' are not NaN
    condition = df['Birth Year'].isna() & df['Age at Injury'].notna() & df['Accident Date'].notna()
    # Replace missing 'Birth Year' with the difference between 'Accident Date' year and 'Age at Injury'
    df.loc[condition, 'Birth Year'] = df.loc[condition, 'Accident Date'].dt.year - df.loc[condition, 'Age at Injury']

    # Replace na with 0
    df['IME-4 Count'] = df['IME-4 Count'].fillna(0)

    # Apply transformations
    df['Average Weekly Wage'] = df['Average Weekly Wage'].apply(lambda x: np.log10(x) if x > 0 else np.nan)
    df['IME-4 Count'] = df['IME-4 Count'].apply(lambda x: np.sqrt(x) if x > 0 else 0)

    # Feature Engineering
    # Known date or not
    date_columns = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date', 'Age at Injury', 'Birth Year']
    for col in date_columns:
        df[f"Known {col}"] = df[col].notna().astype(int)

    # Create columns for year, month, day, and day of the week
    date_columns = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']
    extract_dates_components([df], date_columns)

    # Calculate Time Differences
    dist_dates([df])

    # Apply the function to the 'Accident Date' column
    df['Accident_Season'] = df['Accident Date'].apply(get_season)

    # Flag public holidays
    df = flag_public_holiday_accidents(df, 'Accident Date')

    # Flag weekends
    df = flag_weekend_accidents(df, 'Accident Date')

    # Risk
    Risk_map = {
        "HEALTH CARE AND SOCIAL ASSISTANCE": 1,  # Medium risk
        "PUBLIC ADMINISTRATION": 0,             # Low risk
        "RETAIL TRADE": 1,                      # Medium risk
        "TRANSPORTATION AND WAREHOUSING": 2,    # High risk
        "EDUCATIONAL SERVICES": 0,              # Low risk
        "MANUFACTURING": 2,                     # High risk
        "CONSTRUCTION": 2,                      # High risk
        "ACCOMMODATION AND FOOD SERVICES": 1,   # Medium risk
        "ADMINISTRATIVE AND SUPPORT AND WASTE MANAGEMENT AND REMEDIAT": 1, # Medium risk
        "WHOLESALE TRADE": 1,                   # Medium risk
        "OTHER SERVICES (EXCEPT PUBLIC ADMINISTRATION)": 0, # Low risk
        "ARTS, ENTERTAINMENT, AND RECREATION": 1,           # Medium risk
        "PROFESSIONAL, SCIENTIFIC, AND TECHNICAL SERVICES": 0, # Low risk
        "INFORMATION": 0,                        # Low risk
        "REAL ESTATE AND RENTAL AND LEASING": 0, # Low risk
        "FINANCE AND INSURANCE": 0,              # Low risk
        "UTILITIES": 2,                          # High risk
        "AGRICULTURE, FORESTRY, FISHING AND HUNTING": 2, # High risk
        "MINING": 2,                             # High risk
        "MANAGEMENT OF COMPANIES AND ENTERPRISES": 0, # Low risk
        "nan": 1 # Medium Risk
    }
    df["Risk_Level"] = df["Industry Code Description"].map(Risk_map)

    # Carrier Type
    df['Carrier Type'] = df['Carrier Type'].replace({'5C. SPECIAL FUND - POI CARRIER WCB MENANDS': '5D. SPECIAL FUND - UNKNOWN'})
    df['Carrier Type'] = df['Carrier Type'].replace({'5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)': '5D. SPECIAL FUND - UNKNOWN'})

    # Replace 'U' with 'Unknown'
    df['Gender'] = df['Gender'].replace({'U': 'Unknown'})
    # Replace 'X' with 'Unknown'
    df['Gender'] = df['Gender'].replace({'X': 'Unknown'})

    # Feature Encoding

    # Columns Selection

