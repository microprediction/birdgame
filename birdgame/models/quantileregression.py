from birdgame.trackers.trackerbase import TrackerBase
import numpy as np
from river import linear_model, optim
from river import preprocessing
    

class QuantileRegressionRiverTracker(TrackerBase):
    """
    A model that tracks the dove location using Quantile regression on stream learning.

    Parameters
    ----------
    horizon : int
        The "look-ahead" in time after which the recorded data becomes valid for updating.
    """

    def __init__(self, horizon=10):
        super().__init__(horizon)
        self.current_x = None
        self.miss_count = 0

        # Initialize river models dictionary
        self.models = {}
        self.lr = 0.005
        for i, alpha in enumerate([0.05, 0.5, 0.95]):
            scale = preprocessing.StandardScaler()

            # you can optimize learning rate or use other optimizer (RMSProp, ...)
            learn = linear_model.LinearRegression(
                intercept_lr=0,
                optimizer=optim.SGD(self.lr),
                loss=optim.losses.Quantile(alpha=alpha)
            )

            model = scale | learn

            self.models[f"q {alpha:.2f}"] = preprocessing.TargetStandardScaler(regressor=model)

    def tick(self, payload):
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

            ### (optional idea)
            # Get the predicted quantile values from the models
            if "q 0.05" in self.models:
                y_lower = self.models["q 0.05"].predict_one({"x": prev_x})
                y_upper = self.models["q 0.95"].predict_one({"x": prev_x})

                # Check if observed value `x` is between the predicted quantiles
                if y_lower <= x <= y_upper:
                    prediction_error = 0  # prediction is within bounds
                    # idea: learn two time when prediction is within bounds
                    for i, alpha in enumerate([0.05, 0.5, 0.95]):
                        self.models[f"q {alpha:.2f}"].learn_one({"x": prev_x}, x)
                else:
                    prediction_error = 1  # prediction is outside bounds
            ###

            # River learn_one (online learning)
            for i, alpha in enumerate([0.05, 0.5, 0.95]):
                self.models[f"q {alpha:.2f}"].learn_one({"x": prev_x}, x)

            self.count += 1

    def predict(self):
        """
        Return a dictionary representing the best guess of the distribution
        modeled as a Gaussian distribution.
        """
        x_mean = self.current_x
        components = []

        if "q 0.05" in self.models:
            # Quantile regression prediction 5%, 50% and 95%
            y_lower = self.models["q 0.05"].predict_one({"x": self.current_x})
            y_mean = self.models["q 0.50"].predict_one({"x": self.current_x})
            y_upper = self.models["q 0.95"].predict_one({"x": self.current_x})

            loc = y_mean
            scale = np.abs((y_upper - y_lower)) / 3.289707253902945    # 3.289707253902945 = (norm.ppf(0.95) - norm.ppf(0.05))
            scale = max(scale, 1e-6)
        else:
            loc = x_mean
            scale = 1.0

        components = {
            "density": {
                "type": "builtin",
                "name": "norm",
                "params": {"loc": loc, "scale": scale}
            },
            "weight": 1
        }

        prediction_density = {
            "type": "mixture",
            "components": [components]
        }
        return prediction_density