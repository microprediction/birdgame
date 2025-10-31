"""
Constants used across the birdgame package.
(may be updated in future releases)
"""

# This constant defines the forecast horizon in seconds
HORIZON = 3

GAME_PARAMS = {
    "investment_fraction": 0.00001,  # fraction of wealth invested each tick
    "inflation_bps": 1,            # inflation in basis points
    "initial_wealth": 1000,        # starting wealth per player
}
