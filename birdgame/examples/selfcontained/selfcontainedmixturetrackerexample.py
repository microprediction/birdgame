import typing
import numpy as np
from birdgame.stats.fewvar import FEWVar
import math
from birdgame.datasources.remotetestdata import remote_test_data_generator


# A fully self-contained example of a model that fits a mixture of two Gaussian distributions
# Quarantine is manually done


class SelfContainedMixtureTrackerExample:
    """
    A model that fits a mixture of two Gaussian distributions, one capturing the core
    distribution and another with a larger variance to capture the tails.

    Parameters
    ----------
    fading_factor : float
        Parameter controlling how quickly older data is de-emphasized in variance estimation.
    horizon : int
        The "look-ahead" in time after which the recorded data becomes valid for updating.
    """

    def __init__(self, fading_factor=0.0001, horizon=10):
        self.fading_factor = fading_factor
        self.horizon = horizon
        self.current_x = None
        self.ewa_dx_core = FEWVar(fading_factor=fading_factor)
        self.ewa_dx_tail = FEWVar(fading_factor=fading_factor)
        self.quarantine = []
        self.count = 0
        self.weights = [0.95, 0.05]  # Heavily weight the core distribution

    def tick(self, payload):
        """
        Ingest a new record (payload), store it internally, and update the
        estimated Gaussian mixture model.

        The core distribution captures regular variance, while the tail distribution
        captures extreme deviations.

        Parameters
        ----------
        payload : dict
            Must contain 'time' (int/float) and 'dove_location' (float).
        """

        x = payload['dove_location']
        t = payload['time']
        self.quarantine.append((t + self.horizon, x))
        self.current_x = x

        valid = [(j, (ti, xi)) for (j, (ti, xi)) in enumerate(self.quarantine) if ti <= t]

        if valid:
            prev_ndx, (ti, prev_x) = valid[-1]
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

            self.quarantine = self.quarantine[:prev_ndx]
            self.count += 1

    def predict(self):
        """
        Return a dictionary representing the best guess of the distribution,
        modeled as a mixture of two Gaussians.
        """
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

            components.append({
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {"loc": x_mean, "scale": x_std}
                },
                "weight": self.weights[i]
            })

        prediction_rec = {
            "type": "mixture",
            "components": components
        }
        return prediction_rec


if __name__ == '__main__':
    tracker = SelfContainedMixtureTrackerExample()
    gen = remote_test_data_generator()
    for payload in gen:
        tracker.tick(payload)
        pdf = tracker.predict()
        if tracker.count > 100:
            break

    from pprint import pprint

    pprint(pdf)
