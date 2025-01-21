import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')  # Use the non-graphical Agg backend
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load the data from a CSV file."""
    return pd.read_csv(file_path)

def filter_years(data, start_year=2022, end_year=2024):
    """Filter the data to include only the specified year range."""
    return data[(data['Yrm'] >= start_year) & (data['Yrm'] <= end_year)]

def group_and_aggregate(data):
    """Group the data by year, plot, and DOY, and calculate mean green leaf length."""
    return data.groupby(['Yrm', 'Plot', 'doy'])['gl'].mean().reset_index()

def smooth_data(grouped_data, window_length=21, polyorder=3):
    """Smooth the data using interpolation and Savitzky-Golay filter."""
    smoothed_data = []
    for (year, plot), group in grouped_data.groupby(['Yrm', 'Plot']):
        doy = group['doy'].values
        gl = group['gl'].values

        # Interpolate
        interp_func = interp1d(doy, gl, kind='linear', fill_value="extrapolate")
        new_doy = np.arange(doy.min(), doy.max() + 1)
        new_gl = interp_func(new_doy)

        # Apply Savitzky-Golay filter
        smoothed_gl = savgol_filter(new_gl, window_length=window_length, polyorder=polyorder, mode='interp')

        smoothed_data.append(pd.DataFrame({'Yrm': year, 'Plot': plot, 'doy': new_doy, 'gl': smoothed_gl}))
    
    return pd.concat(smoothed_data, ignore_index=True)

def estimate_senescence(smoothed_data):
    """Estimate senescence day based on the steepest decline after the maximum."""
    senescence_days = []
    for (year, plot), group in smoothed_data.groupby(['Yrm', 'Plot']):
        doy = group['doy'].values
        gl = group['gl'].values

        max_index = np.argmax(gl)
        post_max_doy = doy[max_index:]
        post_max_gl = gl[max_index:]

        if len(post_max_doy) < 2:
            continue

        gl_derivative = np.gradient(post_max_gl, post_max_doy)
        senescence_day = post_max_doy[np.argmin(gl_derivative)]

        senescence_days.append({'Year': year, 'Plot': plot, 'Senescence_DOY': senescence_day})
    
    return pd.DataFrame(senescence_days).dropna()

def add_senescence_to_data(data, senescence_df):
    """Add the senescence start day as a new column in the original data."""
    senescence_mapping = {
        (row['Year'], row['Plot']): row['Senescence_DOY']
        for _, row in senescence_df.iterrows()
    }
    data['senescence_start_doy'] = data.apply(
        lambda row: senescence_mapping.get((row['Yrm'], row['Plot']), None),
        axis=1
    )
    return data

def add_senescence_status(data):
    """
    Add a 'senescence_triggered' column.
    The column is 1 if the current DOY is greater than or equal to the senescence_start_doy,
    otherwise 0.
    """
    data['senescence_triggered'] = data.apply(
        lambda row: 1 if row['doy'] >= row['senescence_start_doy'] else 0, axis=1
    )
    return data

def add_lat_long(data):
    """
    Add latitude and longitude to the data based on the site.
    """
    site_coordinates = {
        'TL': (68.66109, -149.37047),
        'SG': (69.4223, -148.6900),
        'CF': (67.2534, -150.1795)
    }
    data['Site'] = data['Plot'].str.extract(r'([A-Za-z]+)')  # Extract site (e.g., TL, SG, CF)
    data['Site'] = data['Site'].str.upper()  # Normalize capitalization (e.g., "Tl" -> "TL")
    data['Latitude'] = data['Site'].map(lambda site: site_coordinates.get(site, (None, None))[0])
    data['Longitude'] = data['Site'].map(lambda site: site_coordinates.get(site, (None, None))[1])
    return data

def calculate_percentage_senesced(data):
    """
    Calculate the percentage of plots that have senesced for each DOY, grouped by site.
    """
    percentage_senesced = (
        data.groupby(['Site', 'Yrm', 'doy'])
        .apply(lambda group: (group['senescence_triggered'].sum() / len(group)) * 100)
        .reset_index(name='Percent_Senesced')
    )
    return percentage_senesced

def plot_senescence(smoothed_data, senescence_df, output_image_path):
    """Plot the smoothed data and mark the senescence day."""
    unique_years = smoothed_data['Yrm'].unique()
    fig, axes = plt.subplots(len(unique_years), 1, figsize=(10, 5 * len(unique_years)), sharex=True)

    if len(unique_years) == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one year

    for ax, year in zip(axes, unique_years):
        year_data = smoothed_data[smoothed_data['Yrm'] == year]
        senescence_year_data = senescence_df[senescence_df['Year'] == year]
        for plot in year_data['Plot'].unique():
            plot_data = year_data[year_data['Plot'] == plot]
            ax.plot(plot_data['doy'], plot_data['gl'])

            senescence_doy = senescence_year_data[senescence_year_data['Plot'] == plot]['Senescence_DOY']
            if not senescence_doy.empty:
                ax.axvline(x=senescence_doy.values[0], color='red', linestyle='--')

        ax.set_title(f'Year {year} - Senescence Detection (Post-Maximum)')
        ax.set_ylabel('Total Green Leaf Length (cm)')

    plt.xlabel('Day of Year (DOY)')
    plt.tight_layout()
    plt.savefig(output_image_path)

def plot_percentage_senesced(percent_data, output_image_path):
    """
    Plot the percentage of plots senesced over DOY for each site, with one subplot per year.
    """
    unique_years = percent_data['Yrm'].unique()
    fig, axes = plt.subplots(len(unique_years), 1, figsize=(10, 5 * len(unique_years)), sharex=True)

    if len(unique_years) == 1:
        axes = [axes]  # Ensure axes is iterable if there's only one year

    for ax, year in zip(axes, unique_years):
        year_data = percent_data[percent_data['Yrm'] == year]
        for site in year_data['Site'].unique():
            site_data = year_data[year_data['Site'] == site]
            ax.plot(site_data['doy'], site_data['Percent_Senesced'], label=f'Site {site}')

        ax.set_title(f'Year {year} - Percentage Senesced')
        ax.set_ylabel('Percent Senesced (%)')
        ax.legend(title='Site')

    plt.xlabel('Day of Year (DOY)')
    plt.tight_layout()
    plt.savefig(output_image_path)

def save_data(data, output_file_path):
    """Save the updated DataFrame to a CSV file."""
    print(data)
    data.to_csv(output_file_path, index=False)

def main():
    file_path = 'ArcticScience25-LGLdata.csv'
    output_file_path = 'ArcticScience25-LGLdata_with_senescence_2022_2024.csv'
    senescence_plot_path = 'senescence_plot_2022_2024.png'
    percent_plot_path = 'percent_senesced_plot_2022_2024.png'

    # Load and filter data
    data = load_data(file_path)
    filtered_data = filter_years(data, start_year=2022, end_year=2024)

    # Process data
    grouped_data = group_and_aggregate(filtered_data)
    smoothed_data = smooth_data(grouped_data)
    senescence_df = estimate_senescence(smoothed_data)
    updated_data = add_senescence_to_data(filtered_data, senescence_df)

    # Add the senescence triggered column
    updated_data = add_senescence_status(updated_data)

    # Add latitude and longitude
    updated_data = add_lat_long(updated_data)

    # Calculate percentage senesced
    percent_senesced_data = calculate_percentage_senesced(updated_data)

    # Save and plot results
    save_data(updated_data, output_file_path)
    plot_senescence(smoothed_data, senescence_df, senescence_plot_path)
    plot_percentage_senesced(percent_senesced_data, percent_plot_path)

    print(f"Updated file saved at: {output_file_path}")
    print(f"Senescence plot saved at: {senescence_plot_path}")
    print(f"Percentage senesced plot saved at: {percent_plot_path}")

if __name__ == '__main__':
    main()
