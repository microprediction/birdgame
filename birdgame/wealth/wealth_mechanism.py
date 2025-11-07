import math
from birdgame import GAME_PARAMS

def update_wealth(players, likelihoods, params=GAME_PARAMS, wealth_update=True):
    """
    Update each player's wealth using a combination of instantaneous likelihood
    and exponentially weighted moving average (EWMA) of log-likelihood.

    Parameters
    ----------
    players : dict
        Dictionary of player states. Each player must have 'wealth' and optionally 'ewma_logL'.
    likelihoods : dict
        Instantaneous likelihood scores from the current tick for each player.
    params : dict
        Game parameters including 'investment_fraction', 'inflation_bps', "alpha_short", "alpha_long", "w_short" and "ewma_weight".
    wealth_update : bool
        If False, skip the wealth redistribution (used for warming up the EWMA statistics).

    Returns
    -------
    None
        Updates `players` in place.
    """
    valid_likelihoods = {k: v for k, v in likelihoods.items() if v is not None}
    if not valid_likelihoods:
        return

    # --- Update EWMA log-likelihood for each player ---
    for name, likelihood in valid_likelihoods.items():
        player = players[name]
        log_likelihood = math.log(max(likelihood, 1e-12))

        if "ewma_long_logL" not in player:
            # Initialize EWMA with first log-likelihood
            player["ewma_short_logL"] = log_likelihood
            player["ewma_long_logL"] = log_likelihood
        else:
            # Update EWMA
            # Short-term log-likelihood
            player["ewma_short_logL"] = params["alpha_short"] * log_likelihood + (1 - params["alpha_short"]) * player["ewma_short_logL"]
            # Long-term log-likelihood
            player["ewma_long_logL"] = params["alpha_long"] * log_likelihood + (1 - params["alpha_long"]) * player["ewma_long_logL"]

        # Blend short-term log-likelihood and long-term log-likelihood performance
        player["ewma_blend_logL"] = params["w_short"] * player["ewma_short_logL"] + (1 - params["w_short"]) * player["ewma_long_logL"]

    # --- Compute exponential of "ewma_blend_logL" ---
    # ewmas = {name: players[name]["ewma_blend_logL"] for name in valid_likelihoods}
    # max_ewma = max(ewmas.values())
    # rel_ewma = {name: math.exp(ewmas[name] - max_ewma) for name in ewmas} # depreciated
    rel_ewma = {name: math.exp(players[name]["ewma_blend_logL"]) for name in valid_likelihoods}

    # Compute totals for normalization
    total_likelihood = sum(valid_likelihoods.values())
    total_rel_ewma = sum(rel_ewma.values())

    # Skip wealth redistribution if still warming up
    if total_likelihood == 0 or total_rel_ewma == 0 or not wealth_update:
        return

    # Investment phase
    # - For each prediction round, players automatically invest a fraction of their active wealth into the pot.
    # - This amount is subtracted from their active wealth.
    # - Players can skip predictions. Doing so means they cannot lose or gain wealth, as they are not participating in prize distribution.
    pot = 0.0
    for name, player in players.items():
        if name in valid_likelihoods:
            investment = params["investment_fraction"] * player["wealth"]
            player["wealth"] = max(0.0, player["wealth"] - investment) # - Player wealth will never go below 0.
            pot += investment

    # Note: player wealth with 0:
    # Even if your wealth reaches zero, you can still make a comeback.
    # Since your investment will be zero, you won't lose anything more, but you can still earn from the shared pot if your tracker performs well.

    # Inflation adjustment
    # - The total pot is inflated slightly by a game-defined inflation rate.
    pot *= 1 + params["inflation_bps"] / 10000.0

    # Redistribution phase
    # - Once the true dove location is revealed, each prediction is scored using a likelihood function.
    # - The pot is then distributed proportionally based on these likelihood scores.
    # - More accurate predictions earn a larger share of the pot.
    for name in valid_likelihoods:
        instant_share = valid_likelihoods[name] / total_likelihood
        ewma_share = rel_ewma[name] / total_rel_ewma if name in rel_ewma else 0
        # Blend instant and ewma performance
        share = (1 - params["ewma_weight"]) * instant_share + params["ewma_weight"] * ewma_share
        players[name]["wealth"] += pot * share