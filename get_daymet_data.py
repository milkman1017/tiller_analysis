import requests
import pandas as pd
from datetime import datetime
from io import StringIO
from tqdm import tqdm

# Function to fetch Daymet data for specific days within a year
def fetch_daymet_by_days(lat, lon, year, days):
    """
    Fetch Daymet data for a specific latitude, longitude, year, and specific days.
    """
    url = "https://daymet.ornl.gov/single-pixel/api/data"
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    params = {
        "lat": lat,
        "lon": lon,
        "vars": 'T2M',
        "start": start_date,
        "end": end_date,
        "format": "csv"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        daymet_data = pd.read_csv(StringIO(response.text), skiprows=6)  # Adjust skiprows for header lines
        print(daymet_data[daymet_data['yday'].isin(days)])
        return daymet_data[daymet_data['yday'].isin(days)]
    else:
        print(f"Error fetching data for {lat}, {lon} in year {year}: {response.status_code}")
        return None

def create_daymet_dataframe(input_data, save_weather_only=False, weather_output_file="weather_data.csv"):
    """
    Merge input data with Daymet data by fetching Daymet data based on latitude, longitude, year, and DOY.

    Parameters:
    - input_data: DataFrame with columns ['Latitude', 'Longitude', 'Yrm', 'doy'].
    - save_weather_only: If True, save only the weather data to a CSV file without merging.
    - weather_output_file: File path to save the weather-only data if save_weather_only is True.

    Returns:
    - DataFrame: Merged input and Daymet data, or None if save_weather_only is True.
    """
    # Cache for storing fetched Daymet data by (lat, lon, year)
    daymet_cache = {}

    # Initialize list for storing results
    results = []

    # Iterate over unique combinations of Latitude, Longitude, and Year
    for (lat, lon, year), group in tqdm(input_data.groupby(['Latitude', 'Longitude', 'Yrm'])):
        # Extract unique days of the year for the current group
        days = group['doy'].unique()

        # Check if data for this (lat, lon, year) is already fetched
        cache_key = (lat, lon, year)
        if cache_key not in daymet_cache:
            daymet_data = fetch_daymet_by_days(lat, lon, year, days)
            if daymet_data is not None:
                daymet_cache[cache_key] = daymet_data
            else:
                continue
        else:
            daymet_data = daymet_cache[cache_key]

        # Add Daymet data for the specific days in the group
        for _, row in group.iterrows():
            yday = row['doy']

            # Filter Daymet data for the specific day
            day_data = daymet_data[daymet_data['yday'] == yday]
            if not day_data.empty:
                day_data = day_data.copy()
                for col in input_data.columns:
                    day_data[col] = row[col]
                results.append(day_data)

    # Combine all results into a single DataFrame
    if results:
        weather_data = pd.concat(results, ignore_index=True)

        if save_weather_only:
            # Save only the weather data to a CSV file
            weather_data.to_csv(weather_output_file, index=False)
            print(f"Weather data saved to '{weather_output_file}'")
            return None
        else:
            # Merge weather data with input data
            return weather_data
    else:
        print("No Daymet data matched the input criteria.")
        return pd.DataFrame()

# Load the input CSV file
input_file = "ArcticScience25-LGLdata_with_senescence_2022_2024.csv"  # Replace with your actual file path
input_data = pd.read_csv(input_file)

# Create the final Daymet DataFrame
save_weather_only_flag = True  # Set to True to save only the weather data
weather_output_file = "weather_data_only.csv"  # File path for weather-only data

daymet_df = create_daymet_dataframe(input_data, save_weather_only=save_weather_only_flag, weather_output_file=weather_output_file)

if daymet_df is not None:
    output_file = "ArcticScience25-LGLdata_with_weather_data.csv"
    daymet_df.to_csv(output_file, index=False)
    print(f"Daymet data saved to '{output_file}'")
