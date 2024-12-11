import pandas as pd
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
import numpy as np
from sklearn.metrics import f1_score

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
    

def find_duplicate_frequencies_and_map(df, column_name, test_df=None):
    """
    Finds values in a DataFrame column that have the same frequency as others.
    If no values share the same frequency, maps the frequency of each value 
    in the column to a new column with 'Enc ' added as a prefix to the original column name.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column.
    - column_name (str): The name of the column to check for duplicate frequencies.
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

        # Apply the same transformation to test_df if provided
        if test_df is not None:
            test_df[new_column_name] = test_df[column_name].map(value_counts / len(df))


def categorize_impact(impact):
    if impact > 50000:
        return 0 # Low
    elif 1000 <= impact <= 50000:
        return 1 # Medium
    else:
        return 2 # High

def financial_impact(df):
    
    adjusted_dependents = df['Number of Dependents'].replace(0, 1)
    
    financial_impact = df['Average Weekly Wage'] / adjusted_dependents

    df['Financial Impact Category'] = financial_impact.apply(categorize_impact)


def sine_cosine_encoding(df, column, mapping):
    df[f"{column}_Ordinal"] = df[column].map(mapping)
    df[f"{column}_Sin"] = np.sin(2 * np.pi * df[f"{column}_Ordinal"] / 4)
    df[f"{column}_Cos"] = np.cos(2 * np.pi * df[f"{column}_Ordinal"] / 4)
    return df.drop(columns=[f"{column}_Ordinal", column])

def calculate_birth_year(df):
    # Ensure the correct format of 'Birth Year'
    df['Accident Date'] = pd.to_datetime(df['Accident Date'], errors='coerce')

    # Filter the rows where 'Birth Year' is NaN, but 'Age at Injury' and 'Accident Date' are not NaN
    condition = df['Birth Year'].isna() & df['Age at Injury'].notna() & df['Accident Date'].notna()

    # Replace missing 'Birth Year' with the difference between 'Accident Date' year and 'Age at Injury'
    df.loc[condition, 'Birth Year'] = df.loc[condition, 'Accident Date'].dt.year - df.loc[condition, 'Age at Injury']


def save_results_csv(model, features, y_train, y_train_pred, y_val, y_val_pred):
    # Define the model name
    model_name = type(model).__name__

    # Calculate F1 scores
    f1_train = f1_score(y_train, y_train_pred, average='macro')
    f1_val = f1_score(y_val, y_val_pred, average='macro')

    # Get model parameters
    model_params = model.get_params()

    # Create a dictionary of results, including the model name in the third column
    result = {
        'Model Name': model_name,
        'F1 Train': f1_train,
        'F1 Validation': f1_val,
        **model_params, 
        'Feature Group': features,
    }

    # Convert dictionary to DataFrame
    result_df = pd.DataFrame([result])

    # Define the file name (CSV in this case)
    filename = "model_results.csv"

    # Check if the file already exists
    try:
        # If the file exists, append the new results without the header
        existing_df = pd.read_csv(filename)
        result_df.to_csv(filename, index=False, header=False, mode='a')  # Append to existing file
    except FileNotFoundError:
        # If the file does not exist, create a new file and write the header
        result_df.to_csv(filename, index=False)

    print(f"Results added to {filename}")

def NA_imputer(train_df, test_df):

    columns = ["Age at Injury","Average Weekly Wage"]

    imputation_value  = train_df[columns].median()
    for col in columns:
            train_df[col] = train_df[col].fillna(imputation_value[col])
            test_df[col] = test_df[col].fillna(imputation_value[col])

    # Ensure 'Accident Date' is in datetime format
    train_df['Accident Date'] = pd.to_datetime(train_df['Accident Date'], errors='coerce')
    test_df['Accident Date'] = pd.to_datetime(test_df['Accident Date'], errors='coerce')

    # Now apply your logic
    condition = train_df['Birth Year'].isna() & train_df['Age at Injury'].notna() & train_df['Accident Date'].notna()
    train_df.loc[condition, 'Birth Year'] = train_df.loc[condition, 'Accident Date'].dt.year - train_df.loc[condition, 'Age at Injury']

    # Filter the rows where 'Birth Year' is NaN, but 'Age at Injury' and 'Accident Date' are not NaN
    condition = test_df['Birth Year'].isna() & test_df['Age at Injury'].notna() & test_df['Accident Date'].notna()
    # Replace missing 'Birth Year' with the difference between 'Accident Date' year and 'Age at Injury'
    test_df.loc[condition, 'Birth Year'] = test_df.loc[condition, 'Accident Date'].dt.year - test_df.loc[condition, 'Age at Injury']

    train_df.drop('Accident Date',axis=1,inplace=True)
    test_df.drop('Accident Date',axis=1,inplace=True)


def create_new_features(train_df, test_df):

    median_wage = train_df['Average Weekly Wage'].median()
    train_df['Relative_Wage'] = np.where(train_df['Average Weekly Wage'] > median_wage, 1,0) #('Above Median', 'Below Median')
    test_df['Relative_Wage'] = np.where(test_df['Average Weekly Wage'] > median_wage, 1,0) #('Above Median', 'Below Median')

    financial_impact(train_df)
    financial_impact(test_df)

    age_bins = [0, 25, 40, 55, 70, 100]
    age_labels = [0,1,2,3,4] #['Young', 'Mid-Age', 'Experienced', 'Senior', 'Elderly']
    train_df['Age_Group'] = pd.cut(
    train_df['Age at Injury'], bins=age_bins, labels=age_labels, right=False
    ).cat.codes
    test_df['Age_Group'] = pd.cut(
        test_df['Age at Injury'], bins=age_bins, labels=age_labels, right=False
    ).cat.codes


def target_decoder():
    class_mapping = {
        0:'1. CANCELLED', 
        1:'2. NON-COMP',
        2:'3. MED ONLY', 
        3:'4. TEMPORARY',
        4:'5. PPD SCH LOSS', 
        5:'6. PPD NSL', 
        6:'7. PTD', 
        7:'8. DEATH'
    }
    return np.array(list(class_mapping.values()))


def version_control():

    file_path = 'version.txt'

    try:
        with open(file_path, 'r') as file:
            count = int(file.read().strip())
    except FileNotFoundError:
        count = 0

    count += 1

    with open(file_path, 'w') as file:
        file.write(str(count))

    return count

def custom_trial_dirname(trial, model_name):
    return f"./GridSearch/{model_name}/trial_{trial.trial_id}"

def float_to_int(df, columns):
    for col in columns:
        df[col] = df[col].astype('Int64')