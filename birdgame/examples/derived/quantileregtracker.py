from birdgame.trackers.trackerbase import TrackerBase
from birdgame import HORIZON
import numpy as np

class QuantileRegressionConstants:
    WARMUP_CUTOFF = 0

try:
    from river import linear_model, optim
    from river import preprocessing
    using_river = True
except ImportError:
    print('To run this example you need to pip install river')
    using_river = False

if using_river:

    class QuantileRegressionRiverTracker(TrackerBase):
        """
        A model that tracks the dove location using Quantile regression on stream learning.

        Parameters
        ----------
        horizon : int
            The prediction horizon in seconds (how far into the future predictions should be made).
        warmup : int
            The number of ticks taken to warm up the model (wealth does not change during this period).
        """

        def __init__(self, horizon=HORIZON):
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

            self.warmup_cutoff = QuantileRegressionConstants.WARMUP_CUTOFF
            self.tick_count = 0

        def tick(self, payload, performance_metrics):
            """
            Ingest a new record (payload), store it internally and update the model.
            
            Function signature can also look like tick(self, payload) since performance_metrics 
            is an optional parameter.

            Parameters
            ----------
            payload : dict
                Must contain 'time' (int/float) and 'dove_location' (float).
            performance_metrics : dict (is optional)
                Dict containing 'wealth', 'likelihood_ewa', 'recent_likelihood_ewa'.
            """
            # # To see the performance metrics on each tick
            # print(f"performance_metrics: {performance_metrics}")

            # # Can also trigger a warmup by checking if a performance metric drops below a threshold
            # if performance_metrics['recent_likelihood_ewa'] < 1.1:
            #     self.tick_count = 0

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

            self.tick_count += 1

        def predict(self):
            """
            Return a dictionary representing the best guess of the distribution
            modeled as a Gaussian distribution.

            If the model is in the warmup period, return None.
            """
            # Check if the model is warming up
            if self.tick_count < self.warmup_cutoff:
                return None

            x_mean = self.current_x
            components = []

            if "q 0.05" in self.models:
                # Quantile regression prediction 5%, 50% and 95%
                y_lower = self.models["q 0.05"].predict_one({"x": self.current_x})
                y_mean = self.models["q 0.50"].predict_one({"x": self.current_x})
                y_upper = self.models["q 0.95"].predict_one({"x": self.current_x})

                loc = x_mean #y_mean
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
        
else:
    QuantileRegressionRiverTracker = None


if __name__ == '__main__':
    live=False # Set to True to use live streaming data; set to False to use data from a CSV file
    QuantileRegressionConstants.USE_THREADING = live
    tracker = QuantileRegressionRiverTracker()
    tracker.test_run(
        live=live,
        step_print=10000 # Print the score and progress every 1000 steps
    )