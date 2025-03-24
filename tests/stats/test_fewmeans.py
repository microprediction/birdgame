import math
import numpy as np
from birdgame.stats.fewmeans import FEWMeans
from birdgame.stats.fewmean import FEWMean

def test_jumps():
    """
    Use Brownian motion with occasional jumps to evaluate how FEWMeans handles
    non-stationary data with sudden changes, and compare it to a single FEWMean.

    We'll:
    - Generate a time series of length N.
    - The base series evolves via Gaussian increments (simulating Brownian motion).
    - At random intervals, we add a large "jump" to mimic abrupt changes.
    - We'll track the performance of FEWMeans and a single FEWMean in predicting
      the next value, comparing their mean squared errors.
    """

    np.random.seed(42)
    N = 1000
    jump_probability = 0.02   # 2% chance of a jump at each step
    jump_size_mean = 10.0     # Average jump size
    increment_std = 1.0        # Std dev of Brownian increments

    # Generate Brownian motion with jumps
    values = [0.0]
    for i in range(1, N):
        increment = np.random.randn() * increment_std
        new_val = values[-1] + increment

        if np.random.rand() < jump_probability:
            # Add a jump
            jump = np.random.randn() * jump_size_mean
            new_val += jump

        values.append(new_val)

    # Initialize FEWMeans with multiple fading factors
    fm = FEWMeans(fading_factors=[0.01, 0.05, 0.1])

    # Initialize a single FEWMean (e.g., medium-speed adaptation)
    single_mean = FEWMean(fading_factor=0.05)

    fewmeans_errors = []
    singlemean_errors = []

    # We'll start measuring errors after the first update
    for i, val in enumerate(values):
        # Predict before update
        if i > 0:
            # FEWMeans prediction
            fm_pred = fm.get()
            fm_error = (val - fm_pred)**2
            fewmeans_errors.append(fm_error)

            # Single FEWMean prediction
            sm_pred = single_mean.get()
            sm_error = (val - sm_pred)**2
            singlemean_errors.append(sm_error)

        # Update both estimators
        fm.update(val)
        single_mean.update(val)

    # Compute mean squared errors
    fm_mse = np.mean(fewmeans_errors)
    sm_mse = np.mean(singlemean_errors)

    assert fm_mse < sm_mse

if __name__ == "__main__":
    test_jumps()