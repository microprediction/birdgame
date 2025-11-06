"""
Constants used across the birdgame package.
(may be updated in future releases)
"""

# This constant defines the forecast horizon in seconds
HORIZON = 3

GAME_PARAMS = {
    "investment_fraction": 0.0001,  # fraction of wealth invested each tick
    "inflation_bps": 1,            # inflation in basis points
    "initial_wealth": 1000,        # starting wealth per player

    # FYI: 1000 steps/ticks ~ 1min
    "alpha_short": 1 / (20 * 1000 / 60),  # (20 seconds) Smoothing factor for exponentially weighted moving average of short-term log-likelihood 
    "alpha_long": 1 / (20 * 1000),        # (20 minutes) Smoothing factor for exponentially weighted moving average of long-term log-likelihood
    "w_short": 0.5,                       # Weighting factor to blend short-term log-likelihood vs long-term log-likelihood
    "ewma_weight": 1.0,                   # Weighting factor to blend EWMA vs instantaneous likelihood for wealth redistribution
}
