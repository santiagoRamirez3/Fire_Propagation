import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_pbond_vs_time(file_paths, fixed_p_site):
    """
    Plots P_bond vs time for multiple CSV files, filtering by a fixed P_site.

    Parameters:
        file_paths (list of str): List of paths to the CSV files.
        fixed_p_site (float): The fixed P_site value to filter the data.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    
    for file_path in file_paths:
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Filter by the fixed P_site
        filtered_data = data[data['P_site'] == fixed_p_site]
        
        # Plot P_bond vs time
        plt.plot(filtered_data['P_bond'], filtered_data['time'], label=f'{file_path}')
    
    # Add labels, legend, and grid
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('P_bond', fontsize=12)
    plt.title(f'P_bond vs Time (P_site = {fixed_p_site})', fontsize=14)
    plt.legend(title='File Paths', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_avg_time_vs_pbond(file_paths, fixed_p_site, interval_size=0.1):
    """
    Plots the average time for P_bond values in intervals of a given size, 
    filtered by a fixed P_site, from multiple CSV files.

    Parameters:
        file_paths (list of str): List of paths to the CSV files.
        fixed_p_site (float): The fixed P_site value to filter the data.
        interval_size (float): The size of the intervals for P_bond grouping.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    
    for file_path in file_paths:
        # Load the CSV file
        data = pd.read_csv(file_path)
        
        # Filter by the fixed P_site
        filtered_data = data[data['P_site'] == fixed_p_site]
        
        if filtered_data.empty:
            print(f"No data found for P_site = {fixed_p_site} in file {file_path}")
            continue

        # Create bins for P_bond
        min_pbond = filtered_data['P_bond'].min()
        max_pbond = filtered_data['P_bond'].max()
        bins = np.arange(min_pbond, max_pbond + interval_size, interval_size)
        
        # Add a column to the DataFrame indicating the bin each P_bond value falls into
        filtered_data['P_bond_bin'] = pd.cut(filtered_data['P_bond'], bins)
        
        # Group by the bins and compute the average time for each bin
        grouped = filtered_data.groupby('P_bond_bin').agg(
            avg_time=('time', 'mean'),
            bin_midpoint=('P_bond', lambda x: (x.min() + x.max()) / 2)  # Midpoint of the bin
        ).dropna()
        
        # Plot the average time vs the bin midpoints
        plt.plot(grouped['bin_midpoint'], grouped['avg_time'], label=f'{file_path}')
    
    # Add labels, legend, and grid
    plt.xlabel('P_bond (Midpoint of Interval)', fontsize=12)
    plt.ylabel('Average Time', fontsize=12)
    plt.title(f'Average Time vs P_bond (P_site = {fixed_p_site})', fontsize=14)
    plt.legend(title='File Paths', fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    
    # Show the plot
    plt.show()