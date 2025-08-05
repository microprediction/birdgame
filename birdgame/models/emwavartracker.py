from birdgame.trackers.trackerbase import TrackerBase
import math
import numpy as np
from birdgame.stats.fewvar import FEWVar


class EMWAVarTracker(TrackerBase):
    """
    A model that fits a mixture of two Gaussian distributions, one capturing the core
    distribution and another with a larger variance to capture the tails.
    Using EWMA variance

    Parameters
    ----------
    fading_factor : float
        Parameter controlling how quickly older data is de-emphasized in variance estimation.
    horizon : int
        The number of time steps into the future that predictions should be made for.
    warmup : int
        The number of ticks taken to warm up the model (wealth does not change during this period).
    """

    def __init__(self, fading_factor=0.0001, horizon=10, warmup=0):
        super().__init__(horizon)
        self.fading_factor = fading_factor
        self.current_x = None
        self.ewa_dx_core = FEWVar(fading_factor=fading_factor)
        self.ewa_dx_tail = FEWVar(fading_factor=fading_factor)
        self.weights = [0.95, 0.05]  # Heavily weight the core distribution

        self.latest_threat = 0.0

        self.warmup_cutoff = warmup
        self.tick_count = 0

    def tick(self, payload, performance_metrics):
        """
        Ingest a new record (payload), store it internally and update the
        estimated Gaussian mixture model.

        The core distribution captures regular variance, while the tail distribution
        captures extreme deviations.

        Function signature can also look like tick(self, payload) since performance_metrics 
        is an optional parameter.

        Parameters
        ----------
        payload : dict
            Must contain 'time' (int/float) and 'dove_location' (float).
        performance_metrics : dict (is optional)
            Dict containing 'wealth', 'likelihood_ewa', 'recent_likelihood_ewa'
        """

        # # To see the performance metrics on each tick
        # print(f"performance_metrics: {performance_metrics}")

        # # Can trigger a warmup by checking if a performance metric drops below a threshold
        # if performance_metrics['recent_likelihood_ewa'] < 1.1:
        #     self.tick_count = 0

        x = payload['dove_location']
        t = payload['time']
        # falcon_x = payload["falcon_location"]
        # wingspan = payload["falcon_wingspan"]
        self.add_to_quarantine(t, x)
        self.current_x = x
        prev_x = self.pop_from_quarantine(t)

        if prev_x is not None:
            x_change = x - prev_x

            # Winsorize the update for the core estimator to avoid tail effects
            threshold = 2.0 * math.sqrt(self.ewa_dx_core.get() if self.count > 0 else 1.0)
            if threshold > 0:
                winsorized_x_change = np.clip(x_change, -threshold, threshold)
            else:
                winsorized_x_change = x_change
            self.ewa_dx_core.update(winsorized_x_change)

            # Feed the tail estimator with double the real change magnitude
            self.ewa_dx_tail.update(2.0 * x_change)

            self.count += 1

        # # Update threat level: high wingspan + low distance => higher uncertainty
        # dist = abs(falcon_x - x)
        # self.latest_threat = wingspan / (dist + 1e-3)  # Avoid divide-by-zero

        self.tick_count += 1

    def predict(self):
        """
        Return a dictionary representing the best guess of the distribution,
        modeled as a mixture of two Gaussians.

        If the model is in the warmup period, return None.
        """
        # Check if the model is warming up
        if self.tick_count < self.warmup_cutoff:
            return None
            
        # the central value (mean) of the gaussian distribution will be represented by the current value
        x_mean = self.current_x
        components = []

        for i, ewa_dx in enumerate([self.ewa_dx_core, self.ewa_dx_tail]):
            try:
                x_var = ewa_dx.get()
                x_std = math.sqrt(x_var)
            except:
                x_std = 1.0

            if x_std <= 1e-6:
                x_std = 1e-6

            # std = max(std, 1e-6)  # Avoid degenerate std

            # # Scale std based on current threat level (tunable factor 0.2)
            # scaled_std = std * (1.0 + 0.2 * self.latest_threat)

            components.append({
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {"loc": x_mean, "scale": x_std}
                },
                "weight": self.weights[i]
            })

        prediction_density = {
            "type": "mixture",
            "components": components
        }

        return prediction_density