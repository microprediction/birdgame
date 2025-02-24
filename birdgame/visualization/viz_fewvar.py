import numpy as np
import matplotlib.pyplot as plt
from birdgame.stats.fewvar import FEWVar


def visualize_fewvar(fading_factor=0.001, list_variances=[1, 5, 2, 8]):
    """ Function to visualize FEWVar adaptation to changing variance """

    # Generate synthetic data with sudden variance shifts
    np.random.seed(42)
    data = np.concatenate([np.random.normal(0, var, 100) for var in list_variances])

    # Initialize FEWVar
    ewa_dx = FEWVar(fading_factor=fading_factor)

    variances = []
    # Update FEWVar and store the variance estimates
    for x in data:
        ewa_dx.update(x)
        variances.append(ewa_dx.get())

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot data points
    ax1.plot(data, label="Observations", color="gray", alpha=0.6)
    ax1.set_ylabel("Data Values")
    ax1.set_xlabel("Time")
    ax1.legend(loc="upper left")

    # Plot variance estimates
    ax2 = ax1.twinx()
    ax2.plot(variances, label="FEWVar Estimated Variance", color="red", linestyle="--")
    ax2.set_ylabel("Variance Estimate")
    ax2.legend(loc="upper right")

    # Add black dashed lines to separate phases
    phase_endpoints = [100, 200, 300]  # Indices where phases change
    for endpoint in phase_endpoints:
        ax1.axvline(x=endpoint, color='black', linestyle='--')
    
    # Annotating the phases in the graph
    ax1.text(25, min(data), 'Low Variance', color='black', fontsize=12)
    ax1.text(125, min(data), 'High Variance', color='black', fontsize=12)
    ax1.text(225, min(data), 'Medium Variance', color='black', fontsize=12)
    ax1.text(325, min(data), 'Extreme Variance', color='black', fontsize=12)

    plt.title(f"FEWVar Adaptation to Changing Variance (fading_factor={fading_factor})")
    plt.show()


if __name__ == '__main__':
    # Visualize FEWVar on synthetic data with different fading_factor values
    list_fading_factor = [0.1, 0.01, 0.0001]
    for factor in list_fading_factor:
        visualize_fewvar(fading_factor=factor, list_variances=[1, 5, 2, 8])