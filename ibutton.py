import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the non-graphical Agg backend
import matplotlib.pyplot as plt

def process_csv(filepath, skip_rows=14):
    """
    Load a CSV file, skip the first few rows, and process the data:
    - Reset index to make the `Date` part a column.
    - Combine the `Date` (index) with the `Date/Time` column.
    - Convert the combined column into `Date` and `Time` separately.
    - Convert `Date` to day of the year.

    Parameters:
    - filepath (str): Path to the CSV file.
    - skip_rows (int): Number of rows to skip at the start of the file. Defaults to 14.

    Returns:
    - pd.DataFrame: Processed DataFrame.
    """
    # Load the CSV, skipping the comment rows
    df = pd.read_csv(filepath, skiprows=skip_rows, index_col=0)

    # Reset the index to make `Date` part of the columns
    df.reset_index(inplace=True)
    df.rename(columns={"index": "Date"}, inplace=True)  # Ensure the column has a label

    # Combine `Date` and `Date/Time` into a single datetime column
    df['Date/Time Combined'] = df['Date'] + ' ' + df['Date/Time']

    # Split `Date/Time Combined` into `Date` and `Time` columns
    df['Date'] = pd.to_datetime(df['Date/Time Combined']).dt.date
    df['Time'] = pd.to_datetime(df['Date/Time Combined']).dt.time

    # Convert `Date` to day of the year
    df['Day_of_Year'] = pd.to_datetime(df['Date']).dt.day_of_year

    # Drop unnecessary columns
    df.drop(columns=['Date/Time Combined', 'Date/Time'], inplace=True)

    df = df.dropna()

    return df

def process_weather_data(df):

    # Load the CSV

    site_map = {"CF": "Coldfood", "SG": "Sagwon"}

    df['site'] = df['site'].map(site_map)

    # Split timestamp into year, day of year (doy), and time
    df['ts'] = pd.to_datetime(df['ts'])
    df['year'] = df['ts'].dt.year
    df['doy'] = df['ts'].dt.dayofyear
    df['time'] = df['ts'].dt.time

    # Group by site, year, and day of year
    grouped = df.groupby(['site', 'year', 'doy'])

    # Calculate daily statistics
    daily_stats = grouped['temp'].agg(
        tmin='min',
        tmax='max',
        tdiff=lambda x: x.max() - x.min(),
        tavg='mean'
    ).reset_index()

    return daily_stats


def aggregate_data(data, t_base=10):
    """
    Aggregates the input data by calculating:
    - Mean (Tavg), minimum (Tmin), and maximum (Tmax) of 'Value'
    - Difference between Tmin and Tmax (Tdiff)
    - Growing Degree Days (GDD)

    Parameters:
    - data (pd.DataFrame): Input DataFrame with 'Day_of_Year' and 'Value' columns.
    - t_base (float): Base temperature for calculating GDD (default is 10°C).

    Returns:
    - pd.DataFrame: Aggregated DataFrame with 'Day_of_Year', 'Tavg', 'Tmin', 'Tmax', 'Tdiff', and 'GDD'.
    """
    # Group by 'Day_of_Year' and compute stats
    aggregated_df = data.groupby('Day_of_Year')['Value'].agg(
        Tavg='mean',  # Average temperature
        Tmin='min',   # Minimum temperature
        Tmax='max'    # Maximum temperature
    ).reset_index()

    # Calculate Tdiff (difference between Tmax and Tmin)
    aggregated_df['Tdiff'] = aggregated_df['Tmax'] - aggregated_df['Tmin']

    # Calculate GDD (Growing Degree Days)
    aggregated_df['GDD'] = ((aggregated_df['Tmax'] + aggregated_df['Tmin']) / 2) - t_base
    # Set GDD to 0 if it's negative
    aggregated_df['GDD'] = aggregated_df['GDD'].apply(lambda x: max(x, 0))

    return aggregated_df

def plot(aggregated_df, weather_df, tussock_label="Tussock", air_max_label="Max Air", air_min_label="Min Air"):
    """
    Plot temperature data from the aggregated DataFrame and weather data.

    Parameters:
    - aggregated_df (pd.DataFrame): DataFrame containing tussock temperature data.
    - weather_df (pd.DataFrame): DataFrame containing air temperature data.
    - tussock_label (str): Label for the tussock temperature in the legend.
    - air_max_label (str): Label for the max air temperature in the legend.
    - air_min_label (str): Label for the min air temperature in the legend.

    Returns:
    - None: Saves the plot as a PNG file.
    """
    # Merge on Day_of_Year / doy
    merged_df = pd.merge(
        aggregated_df, 
        weather_df, 
        left_on='Day_of_Year', 
        right_on='doy', 
        suffixes=('_tussock', '_air')
    )

    print(merged_df)

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(merged_df['Day_of_Year'], merged_df['Tavg'], label=tussock_label)
    plt.plot(merged_df['Day_of_Year'], merged_df['tmax'], label=air_max_label, linestyle='--', linewidth=1)
    plt.plot(merged_df['Day_of_Year'], merged_df['tmin'], label=air_min_label, linestyle=':', linewidth=1)
    plt.plot(merged_df['Day_of_Year'],merged_df['tavg'], label='Air avg Temp', linestyle='-.', linewidth=1)

    # Customize plot
    plt.title("Temperature Comparison: Tussock vs Air", fontsize=16)
    plt.xlabel("Day of Year", fontsize=14)
    plt.ylabel("Temperature (°C)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # Save the plot
    plt.tight_layout()
    plt.savefig('daily_ibutton_vs_air_max_min.png')



def main():
    year = 2022
    site = 'Sagwon'
    tussock = 'SG-SG-2C'

    ibutton_data_path = f'iButtons/{site}/{tussock}.csv'

    print('Cleaning iButton data...')
    cleaned_ibutton_data = process_csv(ibutton_data_path)

    print('Aggregating iButton data...')
    aggregated_ibutton_df = aggregate_data(cleaned_ibutton_data)

    print('Loading weather data...')
    weather_data = pd.read_csv('CFetSG_WSdata.csv')
    weather_data = process_weather_data(weather_data)
    print(weather_data)

    # Filter weather data for the matching year and site
    weather_data = weather_data[weather_data['year'] == year]
    print(weather_data)

    print('Plotting data...')
    plot(aggregated_ibutton_df, weather_data)

if __name__ == "__main__":
    main()
