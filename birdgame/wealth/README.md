# Wealth Distribution Mechanism

We've just rolled out an **important update** to the wealth distribution logic in *Birdgame*.  
This new mechanism aims to make the game **more stable**, **less luck-driven** and **more reflective of consistent performance** across long time.

## What Changed

Previously, each player's wealth was updated **purely based on their instantaneous likelihood score** for each prediction tick.  
While this was simple and responsive, it also made the results quite **volatile**, wealth could fluctuate rapidly, sometimes due to short-term randomness rather than consistent performance.

To improve stability and reward consistent performance, we've included a **timescale exponential weighting system**.

## The New Formula

The function `update_wealth()` now combines **instantaneous accuracy** with an **exponentially weighted moving average (EWMA)** of your model's log-likelihood.

### Short-term vs Long-term performance

Each player now maintains two smoothed statistics:

- **Short-term EWMA** (`ewma_short_logL`) — tracks recent performance over roughly **20 seconds**  
- **Long-term EWMA** (`ewma_long_logL`) — tracks stability over roughly **20 minutes**

These are blended into a single **performance metric**:

```python
ewma_blend_logL = w_short * ewma_short_logL + (1 - w_short) * ewma_long_logL
```

By default:
```python
w_short = 0.5
```

## Wealth Update Phase

At each tick:

1. **Investment phase**  
   Every active player invests a small fraction of their wealth (`investment_fraction = 0.0001`) into the shared pot.

2. **Inflation phase**  
   The pot is slightly inflated (`inflation_bps = 1`) to represent economic growth.

3. **Redistribution phase**  
   The pot is redistributed based on a weighted mix of:
   - **Instantaneous likelihood** (recent accuracy)
   - **EWMA-based performance** (ewma log-likelihood performance)

The combination is controlled by:

```python
instant_share = likelihood / total_likelihood_all_players
ewma_share = rel_ewma / total_rel_ewma_all_players
share = (1 - ewma_weight) * instant_share + ewma_weight * ewma_share
```

where:
- ```ewma_weight = 1.0``` → fully rely on ewma log-likelihood performances
- ```ewma_weight = 0.0``` → purely short-term reactions

## Current Game Parameters

Check `constants.py`

```python
GAME_PARAMS = {
    "investment_fraction": 0.0001,  # fraction of wealth invested each tick
    "inflation_bps": 1,            # inflation in basis points
    "initial_wealth": 1000,        # starting wealth per player

    # FYI: 1000 steps/ticks ~ 1min
    "alpha_short": 1 / (20 * 1000 / 60),  # (short=20 seconds) Smoothing factor for exponentially weighted moving average of short-term log-likelihood 
    "alpha_long": 1 / (20 * 1000),        # (long=20 minutes) Smoothing factor for exponentially weighted moving average of long-term log-likelihood
    "w_short": 0.5,                       # Weighting factor to blend short-term log-likelihood vs long-term log-likelihood
    "ewma_weight": 1.0,                   # Weighting factor to blend EWMA vs instantaneous likelihood for wealth redistribution
    # When submitted to the platform, the model will be in a warmup phase for at least long=20 minutes.
}
```

**Currently, `"ewma_weight": 1.0`, the redistribution formula relies exclusively on the exponentially weighted moving average of log-likelihoods, excluding the instantaneous likelihood component.**

## Remarks

- When submitted to the platform, the model will be in a warmup phase for at least long=20 minutes to compute full long-term EWMA (`ewma_long_logL`).
- Even if your wealth reaches zero, you can still make a comeback. Since your investment will be zero, you won't lose anything more, but you can still earn from the shared pot if your tracker performs well.


**You can test the wealth distribution mechanism in the notebook [wealth_distribution](https://github.com/microprediction/birdgame/blob/main/birdgame/examples/quickstarters/wealth_distribution.ipynb).**
