import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# Load data from CSV file
def load_data(file_path):
    """Load the x, y, z data from a CSV file."""
    data = pd.read_csv(file_path)
    if not {'P_site', 'P_bond', 'time'}.issubset(data.columns):
        raise ValueError("The CSV file must contain 'P_site', 'P_bond', and 'time' columns.")
    return data

# Create a grid and interpolate data
def create_grid(x, y, z, grid_size=100):
    """Interpolate scattered data onto a grid."""
    xi = np.linspace(x.min(), x.max(), grid_size)
    yi = np.linspace(y.min(), y.max(), grid_size)
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method='cubic')
    return xi, yi, zi

# Compute gradients
def compute_gradients(xi, yi, zi):
    """Compute the gradient of z with respect to x and y on a grid."""
    dz_dx, dz_dy = np.gradient(zi, xi[0, :], yi[:, 0])
    gradient_magnitude = np.sqrt(dz_dx**2 + dz_dy**2)
    return gradient_magnitude

# Normalize gradient magnitude
def normalize_data(data):
    """Normalize data to the range [0, 1]."""
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

# Plot the gradient magnitude heatmap and histogram
def plot_heatmap_and_histogram(xi, yi, normalized_gradient):
    """Plot the gradient magnitude heatmap and its histogram in a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Heatmap
    ax1 = axes[0]
    heatmap = ax1.imshow(normalized_gradient, extent=(xi.min(), xi.max(), yi.min(), yi.max()),
                         origin='lower', cmap='viridis', aspect='auto')
    fig.colorbar(heatmap, ax=ax1, label='Normalized |∇z|')
    ax1.set_title('Gradient Magnitude Heatmap (Normalized)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    # Histogram
    ax2 = axes[1]
    ax2.hist(normalized_gradient.flatten(), bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax2.set_title('Histogram of Normalized Gradient Magnitude')
    ax2.set_xlabel('Normalized |∇z|')
    ax2.set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Main function
def main(file_path):
    # Load data
    data = load_data(file_path)
    x, y, z = data['P_site'].values, data['P_bond'].values, data['time'].values

    # Interpolate data onto a grid
    xi, yi, zi = create_grid(x, y, z)

    # Compute gradient magnitude
    gradient_magnitude = compute_gradients(xi, yi, zi)

    # Normalize gradient magnitude
    normalized_gradient = normalize_data(gradient_magnitude)

    # Plot heatmap and histogram
    plot_heatmap_and_histogram(xi, yi, normalized_gradient)

# Example usage
if __name__ == "__main__":
    file_path = r'data/squared/datos.csv'  # Update with your CSV file path
    main(file_path)
