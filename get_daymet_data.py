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
        "vars": "T2MWET,QV2M,RH2M,T2M_MAX,ALLSKY_SFC_SW_DWN,PS,T2MDEW,WS2M,T2M_MIN,T2M,PRECTOTCORR",
        "start": start_date,
        "end": end_date,
        "format": "csv"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        daymet_data = pd.read_csv(StringIO(response.text), skiprows=6)  # Adjust skiprows for header lines
        return daymet_data[daymet_data['yday'].isin(days)]
    else:
        print(f"Error fetching data for {lat}, {lon} in year {year}: {response.status_code}")
        return None

def create_daymet_dataframe(input_data):
    """
    Merge input data with Daymet data by fetching Daymet data based on latitude, longitude, year, and DOY.
    """
    # Cache for storing fetched Daymet data by (lat, lon, year)
    daymet_cache = {}

    # Initialize list for storing results
    results = []

    # Iterate over unique combinations of Site, Latitude, Longitude, and Year
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
        final_df = pd.concat(results, ignore_index=True)
        return final_df
    else:
        print("No Daymet data matched the input criteria.")
        return pd.DataFrame()

# Load the input CSV file
input_file = "ArcticScience25-LGLdata_with_senescence_2022_2024.csv"  # Replace with your actual file path
input_data = pd.read_csv(input_file)

# Create the final Daymet DataFrame
daymet_df = create_daymet_dataframe(input_data)

# Save the resulting Daymet data to a CSV file
output_file = "ArcticScience25-LGLdata_with_weather_data.csv"
daymet_df.to_csv(output_file, index=False)
print(f"Daymet data saved to '{output_file}'")
