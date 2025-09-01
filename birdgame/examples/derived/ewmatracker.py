from birdgame.trackers.trackerbase import TrackerBase
from birdgame.datasources.livedata import live_data_generator
from pprint import pprint
import math
import numpy as np
from birdgame.stats.fewvar import FEWVar


class EMWAConstants:
    HORIZON = 10
    FADE_FACTOR = 0.0001

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
    """

    def __init__(self):
        super().__init__(EMWAConstants.HORIZON)
        self.fading_factor = EMWAConstants.FADE_FACTOR
        self.current_x = None
        self.ewa_dx_core = FEWVar(fading_factor=EMWAConstants.FADE_FACTOR)
        self.ewa_dx_tail = FEWVar(fading_factor=EMWAConstants.FADE_FACTOR)
        self.weights = [0.95, 0.05]  # Heavily weight the core distribution

    def tick(self, payload, performance_metrics):
        """
        Ingest a new record (payload), store it internally and update the
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

    def predict(self):
        """
        Return a dictionary representing the best guess of the distribution,
        modeled as a mixture of two Gaussians.
        """
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

def example_of_testing_manually():
    # Just an example
    gen = live_data_generator()
    pdf = None
    for payload in gen:
        tracker.tick(payload)
        pdf = tracker.predict()
        if tracker.count > 100:
            break
    pprint(pdf)


if __name__ == '__main__':
    tracker = EMWAVarTracker()
    tracker.test_run(
        live=False, # Set to True to use live streaming data; set to False to use data from a CSV file
        step_print=10000 # Print the score and progress every 1000 steps
    )