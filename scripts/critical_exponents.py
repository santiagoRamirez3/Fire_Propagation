import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

def plot_critical_exponents(data, P_site):
    """
    Plots ln(time) vs ln((P_bond - Pc)/Pc) for a single P_site,
    along with linear regression lines before and after Pc.
    
    Parameters:
    - data: DataFrame containing columns 'P_site', 'P_bond', 'time'.
    - P_site: The specific P_site to analyze and plot.
    """
    # Filter the data for the selected P_site
    group = data[data['P_site'] == P_site]
    
    if group.empty:
        print(f"No data found for P_site={P_site}.")
        return None
    
    # Find Pc (P_bond corresponding to the maximum time)
    max_time_index = group['time'].idxmax()
    print(max_time_index)
    Pc = group.at[max_time_index, 'P_bond']
    group = group.drop(max_time_index)
    
    # Calculate ln(time) and ln((P_bond - Pc)/Pc)
    group['ln_time'] = np.log(group['time'])
    group['ln_x'] = np.log(np.abs((group['P_bond'] - Pc) / Pc))
    
    # Sort by ln_x for consistency
    #group = group.sort_values('ln_x')
    
    # Split the data into two segments: before and after Pc
    before_Pc = group[group['P_bond'] < Pc]
    before_Pc = before_Pc[before_Pc['ln_x'] >= -2]

    after_Pc = group[group['P_bond'] > Pc]
    after_Pc = after_Pc[after_Pc['ln_x'] >= -2]
    
    # Debugging output
    print(f"P_site: {P_site}, Pc: {Pc}")
    print("Data before Pc:")
    print(before_Pc)
    print("Data after Pc:")
    print(after_Pc)
    
    # Check for empty segments
    if before_Pc.empty or after_Pc.empty:
        print(f"Insufficient data for linear regression on P_site {P_site}.")
        return None
    
    # Perform linear regression on both segments
    slope_before, intercept_before, _, _, _ = linregress(before_Pc['ln_x'], before_Pc['ln_time'])
    slope_after, intercept_after, _, _, _ = linregress(after_Pc['ln_x'], after_Pc['ln_time'])
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(before_Pc['ln_x'], before_Pc['ln_time'], label=f'P_site={P_site} before {slope_before}', color='violet', alpha=0.6)
    plt.scatter(after_Pc['ln_x'], after_Pc['ln_time'], label=f'P_site={P_site} after {slope_after}', color='black', alpha=0.6)
    
    # Plot regression lines
    plt.plot(before_Pc['ln_x'], slope_before * before_Pc['ln_x'] + intercept_before, 
             color='red', label='Regression (Before Pc)')
    plt.plot(after_Pc['ln_x'], slope_after * after_Pc['ln_x'] + intercept_after, 
             color='blue', label='Regression (After Pc)')
    
    # Configure the plot
    plt.xlabel('ln((P_bond - Pc) / Pc)')
    plt.ylabel('ln(time)')
    plt.title(f'Critical Exponents Analysis for P_site={P_site}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# Example Usage
# Load the dataset
file_path = r'data/squared/datos_alta_resolucion.csv'
data = pd.read_csv(file_path)

# Specify the P_site to plot
selected_P_site = 1  # Replace with the desired P_site value
plot_critical_exponents(data, selected_P_site)

#plt.plot(data[data['P_site'] == 1]['P_bond'], data[data['P_site'] == 1]['time'])
#plt.show()
