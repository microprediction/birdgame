import math
import numpy as np
from collections import deque
from birdgame.trackers.trackerbase import TrackerBase
from birdgame.examples.derived.gmmtracker import GMMTracker
from densitypdf import density_pdf
from birdgame.stats.fewvar import FEWVar


class VolScaledGMMTracker(TrackerBase):
    """
    A tracker that composes the long-term shape estimates of GMMTracker with
    a short-term volatility estimate from an exponentially weighted variance (FEWVar).
    """

    def __init__(
        self,
        gmm_tracker=None,
        n_components=2,
        scale_cap=3.0,
        horizon=3,
        batch_size=200,
        burn_in=1000,
        window_len=10000,
        data_shrinkage=0.95,
        fading_factor=0.01
    ):
        """
        Args:
            gmm_tracker (GMMTracker): An instance of GMMTracker configured for
                                      your desired long-term behavior. If None,
                                      we'll create one for you.
            n_components (int): If we create a default GMMTracker, the # of mixture components.
            scale_cap (float): A clamp to avoid extreme blow-ups or near-zero scales.
            horizon (int): The horizon parameter for quarantine logic (and for GMMTracker).
            batch_size (int): The batch size for re-fitting the GMM.
            burn_in (int): The burn-in period for GMM predictions (fallback to mixture otherwise).
            window_len (int): The max number of diffs we store in GMMTracker.
            data_shrinkage (float): Fraction of the mean to remove from data before fitting GMM.
            fading_factor (float): The fading factor for the FEWVar short-term volatility estimate.
        """
        super().__init__(horizon)

        # If the user didn't supply a GMMTracker, create one with these defaults.
        if gmm_tracker is None:
            gmm_tracker = GMMTracker(
                n_components=n_components,
                horizon=horizon,
                batch_size=batch_size,
                burn_in=burn_in,
                window_len=window_len,
                data_shrinkage=data_shrinkage
            )

        self.gmm_tracker = gmm_tracker

        # Instead of a deque of short diffs, we track an exponentially weighted variance
        self.short_var = FEWVar(fading_factor=fading_factor)

        # We can clamp extreme scale factors
        self.scale_cap = scale_cap

    def tick(self, payload, performance_metrics):
        """
        Pass the incoming data point to the GMMTracker as usual, and also
        feed the difference to our FEWVar for short-term variance.
        """
        # 1) Update the underlying GMMTracker
        self.gmm_tracker.tick(payload, performance_metrics)

        # 2) Our own short-horizon difference logic
        x = payload["dove_location"]
        t = payload["time"]
        self.add_to_quarantine(t, x)
        prev_x = self.pop_from_quarantine(t)

        if prev_x is not None:
            x_change = x - prev_x
            self.short_var.tick(x_change)

    def predict(self):
        """
        Get the mixture distribution from the GMMTracker. Then rescale its standard
        deviations by comparing the short-term volatility (sqrt of FEWVar) to the
        mixture's average scale.
        """
        mixture = self.gmm_tracker.predict()

        # If we're still using fallback (not a mixture or no components), just return as is
        if mixture.get("type") != "mixture" or len(mixture.get("components", [])) == 0:
            return mixture

        # If we haven't accumulated any meaningful variance yet, there's not much to scale
        # For FEWVar, if there's only 1 sample or none, the variance is 0 or None
        short_var = self.short_var.get_var()
        if short_var <= 0:
            return mixture

        short_std = math.sqrt(short_var)

        # Calculate the mixture's weighted average standard deviation
        weights = [comp["weight"] for comp in mixture["components"]]
        stdevs = [comp["density"]["params"]["scale"] for comp in mixture["components"]]
        mixture_avg_std = sum(w * s for w, s in zip(weights, stdevs))

        # If the mixture stdev is extremely small, skip rescaling
        if mixture_avg_std < 1e-9:
            return mixture

        # Ratio: short-term volatility / GMM's average volatility
        scale_factor = short_std / mixture_avg_std

        # Optionally clamp it to avoid extremes
        scale_factor = max(scale_factor, 1.0 / self.scale_cap)
        scale_factor = min(scale_factor, self.scale_cap)

        # Scale each component's std
        scaled_components = []
        for comp in mixture["components"]:
            new_std = comp["density"]["params"]["scale"] * scale_factor
            scaled_components.append({
                "density": {
                    "type": comp["density"]["type"],
                    "name": comp["density"]["name"],
                    "params": {
                        "loc": comp["density"]["params"]["loc"],
                        "scale": float(new_std)
                    }
                },
                "weight": comp["weight"]
            })

        scaled_mixture = {
            "type": "mixture",
            "components": scaled_components
        }

        # Quick validity check
        _ = density_pdf(scaled_mixture, x=0.0)
        return scaled_mixture


if __name__ == '__main__':
    # Example usage: create a VolScaledGMMTracker with default GMMTracker under the hood.
    tracker = VolScaledGMMTracker(
        n_components=2,
        horizon=3,
        batch_size=200,
        burn_in=1000,
        window_len=10000,
        data_shrinkage=0.95,
        scale_cap=3.0,
        fading_factor=0.01  # for FEWVar
    )
    # Test-run method (assuming 'test_run' is from your TrackerBase):
    tracker.test_run(live=False, step_print=200)
