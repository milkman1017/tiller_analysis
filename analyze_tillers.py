import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.signal import argrelextrema

def load_data(filepath):
    """Loads CSV data into a Pandas DataFrame."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Drops unnecessary columns from the DataFrame."""
    return df.drop(columns=['Rep', 'gr', 'tr', 'Yrt'])

def filter_leaves(df):
    """Removes leaves if they have less than half the number of measurements as their tiller or fewer than 4 measurements total."""
    tiller_counts = df.groupby(['Site', 'Plot', 'Ind', 'Tiller', 'Yrm'])['doy'].count().rename('tiller_count').reset_index()
    leaf_counts = df.groupby(['Site', 'Plot', 'Ind', 'Tiller', 'Leaf', 'Yrm'])['doy'].count().rename('leaf_count').reset_index()
    
    merged_counts = leaf_counts.merge(tiller_counts, on=['Site', 'Plot', 'Ind', 'Tiller', 'Yrm'])
    valid_leaves = merged_counts[(merged_counts['leaf_count'] >= (merged_counts['tiller_count'] / 3)) & (merged_counts['leaf_count'] >= 3)]
    
    df = df.merge(valid_leaves[['Site', 'Plot', 'Ind', 'Tiller', 'Leaf', 'Yrm']], on=['Site', 'Plot', 'Ind', 'Tiller', 'Leaf', 'Yrm'])
    
    return df

def interpolate_data(df):
    """Interpolates daily values first, then applies moving average smoothing and calculates standard deviation."""
    interpolated = []
    
    for (site, plot, ind, tiller, leaf, yrm, src), group in df.groupby(['Site', 'Plot', 'Ind', 'Tiller', 'Leaf', 'Yrm', 'Src']):
        group = group.sort_values(by='doy').drop_duplicates(subset='doy').set_index('doy')
        
        # Interpolate daily values
        doy_range = np.arange(group.index.min(), group.index.max() + 1)
        group = group.reindex(doy_range)
        group['gl_original'] = group['gl']  # Store original
        group = group.interpolate()
        
        # Apply moving average smoothing
        group['gl_smoothed'] = group['gl'].rolling(window=14, center=True, min_periods=1).mean()
        
        # Compute rolling standard deviation
        group['gl_std'] = group['gl'].rolling(window=14, center=True, min_periods=1).std()
        
        group[['Site', 'Plot', 'Ind', 'Tiller', 'Leaf', 'Yrm', 'Src']] = site, plot, ind, tiller, leaf, yrm, src
        interpolated.append(group.reset_index())
    
    return pd.concat(interpolated, ignore_index=True)

def plot_data(df, output_dir='plots'):
    """Creates and saves plots of total green length (gl) per tiller against day of year (doy) for each year with subplots per Site-Plot-Ind."""
    os.makedirs(output_dir, exist_ok=True)
    df['Tiller'] = df['Tiller'].str.lower()  # Ensure consistency in plotting
    
    for year, year_group in df.groupby('Yrm'):
        site_plot_ind_combos = list(year_group.groupby(['Site', 'Plot', 'Ind']).groups.keys())
        nrows = (len(site_plot_ind_combos) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(30, 6 * nrows), sharex=True)
        axes = axes.flatten()
        
        for i, (site, plot, ind) in enumerate(site_plot_ind_combos):
            ax = axes[i]
            subgroup = year_group[(year_group['Site'] == site) & (year_group['Plot'] == plot) & (year_group['Ind'] == ind)]
            
            # Ensure gl_smoothed and gl_std exist in subgroup before aggregation
            subgroup['gl_smoothed'] = subgroup['gl_smoothed'].fillna(0)
            subgroup['gl_std'] = subgroup['gl_std'].fillna(0)
            
            tiller_sums_original = subgroup.groupby(['doy', 'Tiller'])['gl_original'].sum().reset_index()
            tiller_sums_interpolated = subgroup.groupby(['doy', 'Tiller'])[['gl_smoothed', 'gl_std']].sum().reset_index()
            
            for tiller, tiller_group in tiller_sums_original.groupby('Tiller'):
                ax.plot(tiller_group['doy'], tiller_group['gl_original'], linestyle='dashed', label=f'Tiller {tiller} (Original)')
            
            for tiller, tiller_group in tiller_sums_interpolated.groupby('Tiller'):
                ax.plot(tiller_group['doy'], tiller_group['gl_smoothed'], linestyle='solid', label=f'Tiller {tiller} (Smoothed)')
                ax.fill_between(tiller_group['doy'], tiller_group['gl_smoothed'] - tiller_group['gl_std'],
                                tiller_group['gl_smoothed'] + tiller_group['gl_std'], alpha=0.2)
            
            ax.set_title(f'Site {site} - Plot {plot} - Tussock {ind} - Year {year}')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel('Total Green Length (gl)')
            ax.set_ylabel('Total Green Length (gl)')
            ax.legend()
            ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        axes[-1].set_xlabel('Day of Year')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/gl_plot_{year}.png')
        plt.close()

def save_final_csv(df, output_filepath='processed_tussock_data.csv'):
    """Saves the final processed DataFrame to a CSV file."""
    df.to_csv(output_filepath, index=False)

def detect_senescence(df):
    """Detects senescence onset using improved methods: Sustained First Derivative Decline, Robust Peak Detection, Second Derivative, Threshold-Based Linear Regression Change, and Cumulative Sum."""
    results = []
    
    for (site, plot, ind, tiller, yrm), group in df.groupby(['Site', 'Plot', 'Ind', 'Tiller', 'Yrm']):
        group = group.sort_values(by='doy').set_index('doy')
        group['gl_smoothed'] = group['gl_smoothed'].fillna(method='ffill').fillna(method='bfill')
        
        # Compute first derivative & smooth it
        group['gl_derivative'] = group['gl_smoothed'].diff()
        group['gl_derivative_smooth'] = group['gl_derivative'].rolling(14, center=False, min_periods=1).mean()
        
        # First Derivative Method: Find the first DOY where decline is sustained for at least 7 days
        decline_days = group[group['gl_derivative_smooth'] < 0].index
        sustained_decline = [day for i, day in enumerate(decline_days[:-13]) if all(d < 0 for d in group['gl_derivative_smooth'].loc[day:day+13])]
        senescence_first_derivative = sustained_decline[0] if sustained_decline else np.nan
        
        # Peak Detection Method: Detect peak and find sustained 10% drop below peak
        peak_doy = group['gl_smoothed'].idxmax()
        peak_value = group['gl_smoothed'].max()
        drop_threshold = peak_value * 0.90
        drop_days = group[(group.index > peak_doy) & (group['gl_smoothed'] < drop_threshold)].index
        sustained_drop = [day for i, day in enumerate(drop_days[:-13]) if all(group['gl_smoothed'].loc[day:day+13] < drop_threshold)]
        senescence_peak_threshold = sustained_drop[0] if sustained_drop else np.nan
        
        # Second Derivative Method: Find the first significant negative peak in second derivative
        group['gl_second_derivative'] = group['gl_derivative_smooth'].diff()
        significant_inflection = group['gl_second_derivative'].idxmin() if not group['gl_second_derivative'].isna().all() else np.nan
        
        # Threshold-Based Linear Regression Change
        rolling_slopes = group['gl_smoothed'].rolling(14, min_periods=14).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=True)
        significant_slope_change = rolling_slopes[rolling_slopes < -0.05].index.min() if not rolling_slopes.isna().all() else np.nan
        
        # Cumulative Sum Change Detection
        group['gl_cumsum'] = (group['gl_smoothed'] - group['gl_smoothed'].expanding().mean()).cumsum()
        cumsum_threshold = group['gl_cumsum'][group['gl_cumsum'] < group['gl_cumsum'].quantile(0.05)].index.min() if not group['gl_cumsum'].isna().all() else np.nan    
        
        # Add senescence onset to all DOYs for consistency
        group['senescence_first_derivative'] = senescence_first_derivative
        group['senescence_peak_threshold'] = senescence_peak_threshold
        group['senescence_second_derivative'] = significant_inflection
        group['senescence_slope_change'] = significant_slope_change
        group['senescence_cumsum'] = cumsum_threshold
        
        results.append(group.reset_index())
    
    return pd.concat(results, ignore_index=True)

def plot_senescence(df, output_dir='senescence_plots'):
    """Plots gl_smoothed vs DOY and overlays predicted senescence start date as vertical lines."""
    os.makedirs(output_dir, exist_ok=True)
    
    for year, year_group in df.groupby('Yrm'):
        site_plot_ind_combos = list(year_group.groupby(['Site', 'Plot', 'Ind']).groups.keys())
        nrows = (len(site_plot_ind_combos) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(30, 6 * nrows), sharex=True)
        axes = axes.flatten()
        
        for i, (site, plot, ind) in enumerate(site_plot_ind_combos):
            ax = axes[i]
            subgroup = year_group[(year_group['Site'] == site) & (year_group['Plot'] == plot) & (year_group['Ind'] == ind)]
            
            for tiller, tiller_group in subgroup.groupby('Tiller'):
                tiller_total_gl = tiller_group.groupby('doy')['gl_smoothed'].sum()
                ax.plot(tiller_total_gl.index, tiller_total_gl.values, linestyle='solid', label=f'Tiller {tiller}')
                
                # Overlay vertical lines for senescence detection
                if not tiller_group['senescence_first_derivative'].isna().all():
                    ax.axvline(tiller_group['senescence_first_derivative'].iloc[0], color='r', linestyle='--', label='First Deriv')
                if not tiller_group['senescence_peak_threshold'].isna().all():
                    ax.axvline(tiller_group['senescence_peak_threshold'].iloc[0], color='b', linestyle='-.', label='Peak Drop')
                if not tiller_group['senescence_second_derivative'].isna().all():
                    ax.axvline(tiller_group['senescence_second_derivative'].iloc[0], color='g', linestyle=':', label='Second Deriv')
                if not tiller_group['senescence_slope_change'].isna().all():
                    ax.axvline(tiller_group['senescence_slope_change'].iloc[0], color='m', linestyle='-.', label='Slope Change')
                if not tiller_group['senescence_cumsum'].isna().all():
                    ax.axvline(tiller_group['senescence_cumsum'].iloc[0], color='y', linestyle=':', label='Cumulative Sum')
            
            ax.set_title(f'Site {site} - Plot {plot} - Tussock {ind} - Year {year}')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel('Smoothed Green Length (gl)')
            ax.legend()
            ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/senescence_{year}.png')
        plt.close()

def save_plot_metadata(df, output_filepath='plot_metadata.csv'):
    """Creates a CSV file with subplot metadata in reading order (left to right)."""
    metadata = []
    
    for year, year_group in df.groupby('Yrm'):
        site_plot_ind_combos = list(year_group.groupby(['Site', 'Plot', 'Ind']).groups.keys())
        for site, plot, ind in site_plot_ind_combos:
            metadata.append([site, plot, ind, year, ""])  # Blank "Onset" column
    
    metadata_df = pd.DataFrame(metadata, columns=['Site', 'Plot', 'Tussock', 'Year', 'Onset'])
    metadata_df.to_csv(output_filepath, index=False)

def plot_smoothed_gl(df, output_dir='smoothed_plots'):
    """Plots smoothed green length (gl) per tiller against day of year (doy) without senescence indicators."""
    os.makedirs(output_dir, exist_ok=True)
    
    for year, year_group in df.groupby('Yrm'):
        site_plot_ind_combos = list(year_group.groupby(['Site', 'Plot', 'Ind']).groups.keys())
        nrows = (len(site_plot_ind_combos) + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(30, 6 * nrows), sharex=True)
        axes = axes.flatten()
        
        for i, (site, plot, ind) in enumerate(site_plot_ind_combos):
            ax = axes[i]
            subgroup = year_group[(year_group['Site'] == site) & (year_group['Plot'] == plot) & (year_group['Ind'] == ind)]
            
            for tiller, tiller_group in subgroup.groupby('Tiller'):
                tiller_total_gl = tiller_group.groupby('doy')['gl_smoothed'].sum()
                ax.plot(tiller_total_gl.index, tiller_total_gl.values, linestyle='solid', label=f'Tiller {tiller}')
            
            ax.set_title(f'Site {site} - Plot {plot} - Tussock {ind} - Year {year}')
            ax.set_xlabel('Day of Year')
            ax.set_ylabel('Smoothed Green Length (gl)')
            ax.legend()
            ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)
            ax.tick_params(axis='both', which='both', labelsize=7, labelbottom=True)
        
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/smoothed_gl_{year}.png')
        plt.close()

def main():
    output_filepath = 'processed_tussock_data.csv'
    
    if os.path.exists(output_filepath):
        df = pd.read_csv(output_filepath)
    else:
        parser = argparse.ArgumentParser(description='Analyze and plot green length data from a CSV file.')
        parser.add_argument('filepath', type=str, help='Path to the CSV file')
        args = parser.parse_args()
    
        df = load_data(args.filepath)
        df = clean_data(df)
        df = filter_leaves(df)
        df = interpolate_data(df)
        # save_final_csv(df)
        # plot_data(df)
    
    df = detect_senescence(df)
    # df.to_csv('processed_tussock_with_senescence.csv', index=False)
    # plot_senescence(df)
    plot_smoothed_gl(df)
    save_plot_metadata(df)

if __name__ == "__main__":
    main()
