import pandas as pd
from scipy import stats
import geopandas as gpd
import matplotlib.pyplot as plt


def TestIndependence(X,y,var,alpha=0.05):        
    dfObserved = pd.crosstab(y,X) 
    chi2, p, dof, expected = stats.chi2_contingency(dfObserved.values)
    dfExpected = pd.DataFrame(expected, columns=dfObserved.columns, index = dfObserved.index)
    if p<alpha:
        result="{0} is IMPORTANT for Prediction".format(var)
    else:
        result="{0} is NOT an important predictor. (Discard {0} from model)".format(var)
    print(result)


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


def find_duplicate_frequencies_and_map(df, column_name):
    """
    Finds values in a DataFrame column that have the same frequency as others.
    If no values share the same frequency, maps the frequency of each value 
    in the column to a new column with 'Enc ' added as a prefix to the original column name.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the column.
    - column_name (str): The name of the column to check for duplicate frequencies.
    
    Returns:
    - None: Prints messages and, if applicable, adds a new column with frequency mappings.
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
