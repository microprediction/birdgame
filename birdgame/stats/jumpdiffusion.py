import random
import numpy as np
import math


# Just here for testing purposes

def jump_diffusion(n_sim, jump_rate, jump_size, drift=0.0, sigma=0.1, epsilon=0.1, vega=2.0):
    """
    Simulate a Brownian motion with two-sided jumps and fat-tailed measurement noise.

    Parameters
    ----------
    n_sim : int
        Number of simulation steps.
    jump_rate : float
        Probability of a jump at any given step.
    jump_size : float
        Magnitude of jumps (added or subtracted).
    drift : float
        Drift term for Brownian motion.
    sigma : float
        Volatility (std dev) of increments.
    epsilon : float
        Noise scale for the gaussian measurement noise.
    vega:  float
        Noise scale for exponential measurement error.

    Returns
    -------
    series : list of floats
        Simulated time series.
    """
    series = []
    x = 0.0
    for _ in range(n_sim):
        # Brownian increment
        increment = random.gauss(drift, sigma)

        # Jump?
        if random.random() < jump_rate:
            jump_direction = 1 if random.random() < 0.5 else -1
            increment += jump_direction * jump_size * np.random.exponential()

        x += increment

        # Fat-tailed noise
        gauss_noise = np.random.randn()
        tail_noise = math.sqrt(np.random.standard_exponential()) * np.sign(np.random.randn())
        y = x + epsilon * gauss_noise + vega * tail_noise
        series.append(y)

    return series