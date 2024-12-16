import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

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
def main(file_path:str, file_name:str) -> None:
    # Load data
    
    data = load_data(file_path + file_name)
    x, y, z = data['P_site'].values, data['P_bond'].values, data['time'].values
    
    # Interpolate data onto a grid
    xi, yi, zi = create_grid(x, y, z)

    # Compute gradient magnitude
    gradient_magnitude = compute_gradients(xi, yi, zi)

    # Normalize gradient magnitude
    normalized_gradient = normalize_data(gradient_magnitude)
    
    # Applyregression using Random Forest
    # Flatten the meshgrid into arrays
    x_flat = xi.ravel()
    y_flat = yi.ravel()
    z_flat = normalized_gradient.ravel()


    # Create a feature matrix and target vector
    features = np.column_stack((x_flat, y_flat))
    target = z_flat

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.05, random_state=40)

    # Train a regression model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    #print("Mean Squared Error:", mean_squared_error(y_test, y_pred))   #Around 0.0009
    
    # Save the model using joblib
    joblib.dump(model, file_path + '3d_regression_model.pkl')
    
    
    # Evaluate the model on the original meshgrid
    original_features = np.column_stack((xi.ravel(), yi.ravel()))
    predicted_Z = model.predict(original_features)  # Predict on the original grid
    predicted_Z_reshaped = predicted_Z.reshape(xi.shape)  # Reshape back to grid shape

    # Plot the original data
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(xi, yi, normalized_gradient, cmap='viridis', edgecolor='none', alpha=0.8)
    ax1.set_title("Original Data")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    
    # Plot the predicted data
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xi, yi, predicted_Z_reshaped, cmap='plasma', edgecolor='none', alpha=0.8)
    ax2.set_title("Predicted Data")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    
    plt.savefig(file_path + 'predicted.png')
    
    # Plot heatmap and histogram
    #plot_heatmap_and_histogram(xi, yi, normalized_gradient)

# Example usage
if __name__ == "__main__":
    file_path = r'data/voronoi/'  # Update with your CSV file path
    file_name = 'datos.csv'
    main(file_path, file_name)

    # Load the model and plot to verify
    
    