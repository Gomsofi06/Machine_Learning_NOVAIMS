import pandas as pd
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

# Visualizations

def plot_categorical_features(df, categorical_features, max_categories=10):
    # Get number of unique values for each feature, sorted for a cleaner layout
    unique_counts = df[categorical_features].nunique().sort_values(ascending=True)
    print("Number of unique values in each feature:")
    print(unique_counts)

    # Determine the number of rows and columns for the subplot grid
    num_features = len(categorical_features)
    num_cols = 3  # Set to 3 columns per row
    num_rows = (num_features + num_cols - 1) // num_cols  # Calculate required rows

    # Set up the figure and axes for subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, num_rows * 5))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each feature on a separate subplot
    for i, feature in enumerate(categorical_features):
        ax = axes[i]
        
        # Get value counts for the current feature
        value_counts = df[feature].value_counts()
        
        # Limit to top categories if needed
        if len(value_counts) > max_categories:
            value_counts = value_counts.head(max_categories)
            ax.set_title(f'Top {max_categories} Categories in {feature}')
        else:
            ax.set_title(f'Distribution of {feature}')
        
        # Create the bar plot on the specified axis
        sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
        ax.set_xlabel('Count')

    # Hide any unused subplots (if num_features is not a perfect multiple of num_cols)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# Function to create bar plots for features with fewer unique values
def plot_value_counts(df, features, max_categories=10, n_cols=2):
        n_features = len(features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, feature in enumerate(features):
            row = idx // n_cols
            col = idx % n_cols
            
            value_counts = df[feature].value_counts()
            if len(value_counts) > max_categories:
                # Keep top categories and group others
                other_count = value_counts[max_categories:].sum()
                value_counts = value_counts[:max_categories]
                value_counts['Others'] = other_count
            
            sns.barplot(x=value_counts.values, y=value_counts.index, ax=axes[row, col])
            axes[row, col].set_title(f'Distribution of {feature}')
            axes[row, col].set_xlabel('Count')
            # Remove empty subplots
        for idx in range(idx + 1, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
            
        plt.tight_layout()
        plt.show()



def plot_cases_by_county(df_cleaned, shapefile_path='NY_counties/Counties.shp'):
    """
    Plots the number of cases by county in New York.

    ----------------------------------------

    Parameters:
    - df_cleaned: DataFrame containing the injury data with a column 'County of Injury'.
    - shapefile_path: Path to the New York counties shapefile.
    """
    # Load New York counties shapefile
    ny_counties = gpd.read_file(shapefile_path)
    
    # Create DataFrame with the number of cases per County
    cases_per_county_df = df_cleaned['County of Injury'].value_counts().reset_index()
    
    # Rename columns
    cases_per_county_df.columns = ['NAME', 'Count']
    
    # Capitalize the first letter of each entry in the 'NAME' column
    cases_per_county_df['NAME'] = cases_per_county_df['NAME'].str.capitalize()
    
    # Merge the cases counts with the counties GeoDataFrame
    ny_counties = ny_counties.merge(cases_per_county_df, on='NAME', how='right')
    
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ny_counties.plot(column='Count', ax=ax, legend=True,
                     legend_kwds={'label': "Number of Cases by County",
                                  'orientation': "horizontal"})
    
    # Remove x and y axis
    ax.set_axis_off()
    
    plt.title('Number of Cases by County')
    plt.show()


# Other

# Define the function that assigns a season based on the exact date
def get_season(date):
    # Extract the month and day from the date
    month_day = (date.month, date.day)
    
    # Define the season boundaries (start of each season)
    winter_start = (12, 21)
    spring_start = (3, 20)
    summer_start = (6, 21)
    fall_start = (9, 23)
    
    # Determine the season based on the month and day
    if (month_day >= winter_start) or (month_day < spring_start):
        return 'Winter'
    elif (month_day >= spring_start) and (month_day < summer_start):
        return 'Spring'
    elif (month_day >= summer_start) and (month_day < fall_start):
        return 'Summer'
    else:
        return 'Fall'
    
def flag_public_holiday_accidents(df, date_column, state='NY'):
    """
    Flags accidents that happened on public holidays in a given state.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the accident data.
    date_column (str): The name of the column containing the accident dates.
    state (str): The state for which public holidays are to be checked (default is 'NY').
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional column flagging public holiday accidents.
    """
    
    # Determine the year range from the 'Accident Date' column
    start_year = df[date_column].dt.year.min()  # Get the earliest year
    end_year = df[date_column].dt.year.max()  # Get the latest year

    # Initialize an empty list to hold all the public holidays for the specified state
    holiday_dates = []

    # Loop through all years from the minimum to the maximum year in the dataset
    for year in range(start_year, end_year + 1):  # Including the end year
        # Get the public holidays for the current year in the specified state
        holidays_in_year = holidays.US(years=year, state=state)
        holiday_dates.extend(holidays_in_year.keys())  # Add holidays for the current year to the list

    # Convert the list of public holiday dates to a pandas datetime format
    holiday_dates = pd.to_datetime(holiday_dates)

    # Create a new column in the dataframe to flag accidents that occurred on public holidays
    df['Holiday_Accident'] = df[date_column].isin(holiday_dates).astype(int)

    return df

def flag_weekend_accidents(df, date_column):
    """
    Flags accidents that happened on weekends (Saturday or Sunday).
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing the accident data.
    date_column (str): The name of the column containing the accident dates.
    
    Returns:
    pd.DataFrame: The input DataFrame with an additional column flagging weekend accidents.
    """
    
    # Create a new column to flag accidents that occurred on weekends
    df['Weekend_Accident'] = df[date_column].dt.dayofweek.isin([5, 6]).astype(int)  # 5=Saturday, 6=Sunday

    return df
    

def find_duplicate_frequencies_and_map(df, column_name, validation_df=None, test_df=None):
    """
    Finds values in a DataFrame column that have the same frequency as others.
    If no values share the same frequency, maps the frequency of each value 
    in the column to a new column with 'Enc ' added as a prefix to the original column name.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column.
    - column_name (str): The name of the column to check for duplicate frequencies.
    - validation_df (pd.DataFrame, optional): Validation DataFrame to apply the same transformation.
    - test_df (pd.DataFrame, optional): Test DataFrame to apply the same transformation.
    
    Returns:
    - None: Prints messages and, if applicable, adds new columns with frequency mappings.
    """
    # Define the new column name
    new_column_name = f"Enc {column_name}"
    
    # Get the counts of each unique value
    value_counts = df[column_name].value_counts()

    # Group values by their frequency
    frequency_groups = value_counts.groupby(value_counts).apply(lambda x: x.index.tolist()).to_dict()
    
    # Check if any frequency group has more than one value (i.e., duplicates)
    duplicate_counts = {freq: values for freq, values in frequency_groups.items() if len(values) > 1}

    if duplicate_counts:
        print("Some values have the same frequency.")
    else:
        print("No values have the same frequency. Mapping frequencies to new column.")
        
        # Map categories to frequency values
        df[new_column_name] = df[column_name].map(value_counts / len(df))
        
        # Apply the same transformation to validation_df if provided
        if validation_df is not None:
            validation_df[new_column_name] = validation_df[column_name].map(value_counts / len(df))

        # Apply the same transformation to test_df if provided
        if test_df is not None:
            test_df[new_column_name] = test_df[column_name].map(value_counts / len(df))



def TestIndependence(X,y,var,alpha=0.05):        
    dfObserved = pd.crosstab(y,X) 
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(var)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)
    print(result)