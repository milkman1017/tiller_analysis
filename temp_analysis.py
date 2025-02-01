import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import pywt
import scipy.signal

def parse_args():
    parser = argparse.ArgumentParser(description="Process a file path.")
    parser.add_argument("--file_path", type=str, help="Path to the input file")
    return parser.parse_args()

def load_and_process_data(file_path):
    df = pd.read_csv(file_path, parse_dates=["ts"])
    df["year"] = df["ts"].dt.year
    df["day_of_year"] = df["ts"].dt.dayofyear
    df["time"] = df["ts"].dt.time
    df["snow"].fillna(0, inplace=True)
    df["temp"] = df["temp"].interpolate()
    return df

def plot_wavelet_transform(df):
    plt.figure(figsize=(12, 6))
    for site in df["site"].unique():
        site_df = df[df["site"] == site]
        temp_values = site_df["temp"].values
        time_values = np.arange(len(temp_values))
        scales = np.arange(1, 365)  # Up to 1-year cycles
        coefficients, frequencies = pywt.cwt(temp_values, scales, 'morl')
        plt.imshow(np.abs(coefficients), aspect='auto', extent=[time_values[0], time_values[-1], scales[-1], scales[0]], cmap='jet')
        plt.colorbar(label='Magnitude')
        plt.xlabel("Time (days)")
        plt.ylabel("Period (days)")
        plt.title(f"Wavelet Transform of Temperature Data for {site}")
        plt.show()

def rolling_lagged_correlation(df, lag=24*200, window=24*50):
    plt.figure(figsize=(12, 6))
    for site in df["site"].unique():
        site_df = df[df["site"] == site].reset_index(drop=True)
        temp_values = site_df["temp"].values
        rolling_corrs = []
        times = []
        for i in range(len(temp_values) - lag - window):
            corr = np.corrcoef(temp_values[i:i+window], temp_values[i+lag:i+lag+window])[0, 1]
            rolling_corrs.append(corr)
            times.append(site_df["ts"].iloc[i])
        
        plt.plot(times, rolling_corrs, label=f"{site}")
    
    plt.xlabel("Time")
    plt.ylabel("Rolling Lagged Correlation (200 days)")
    plt.title("Rolling Lagged Correlation of Temperature Data Across Sites")
    plt.legend()
    plt.show()

def main():
    args = parse_args()
    df = load_and_process_data(args.file_path)
    print(df.head())
    # plot_wavelet_transform(df)
    rolling_lagged_correlation(df)

if __name__ == "__main__":
    main()
