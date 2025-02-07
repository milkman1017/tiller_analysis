import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

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
        plot_data(df)

# Run script with a given CSV file
if __name__ == "__main__":
    main()
