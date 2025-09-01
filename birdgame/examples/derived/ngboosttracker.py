import os
import math
import pandas as pd
import numpy as np
from birdgame.trackers.trackerbase import TrackerBase
import warnings

warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")


class NGBoostConstants:
    HORIZON = 10
    TRAIN_MODEL_FREQUENCY=100
    NUM_DATA_POINTS_MAX=1000
    WINDOW_SIZE = 5

try:
    from ngboost import NGBoost
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor
    using_ngboost = True
except ImportError:
    print('To run this example you need to pip install ngboost scikit-learn')
    using_ngboost = False

if using_ngboost:

    class NGBoostTracker(TrackerBase):
        """
        A model that tracks the dove location using NGBoost.

        Parameters
        ----------
        horizon : int
            The number of time steps into the future that predictions should be made for.
        train_model_frequency : int
            The frequency at which the NGBoost model will be retrained based on the count of observations
            ingested. This determines how often the model will be updated with new data.
        num_data_points_max : int
            The maximum number of data points to use for training the NGBoost model.
        window_size : int
            The number of previous data points (the sliding window size) used to predict the future value
            at the horizon. It defines how many past observations are considered for prediction.
        """

        def __init__(self):
            super().__init__(NGBoostConstants.HORIZON)
            self.current_x = None
            self.last_observed_data = []  # Holds the last few observed data points
            self.x_y_data = []  # Holds pairs of previous and current data points

            self.train_model_frequency = NGBoostConstants.TRAIN_MODEL_FREQUENCY
            self.num_data_points_max = NGBoostConstants.NUM_DATA_POINTS_MAX  # (X.shape[0])
            self.window_size = NGBoostConstants.WINDOW_SIZE  # (X.shape[1])

            # Initialize the NGBoost model
            self.model = NGBoost(Dist=Normal, learning_rate=0.1, n_estimators=50, natural_gradient=True, verbose=False,
                                 random_state=15,
                                 validation_fraction=0.1, early_stopping_rounds=None,
                                 Base=DecisionTreeRegressor(
                                     criterion="friedman_mse",
                                     min_samples_split=2,
                                     min_samples_leaf=1,
                                     min_weight_fraction_leaf=0.0,
                                     max_depth=5,
                                     splitter="best",
                                     random_state=None,
                                 ))

        def tick(self, payload, performance_metrics):
            """
            Ingest a new record (payload), store it internally and update the model.

            Parameters
            ----------
            payload : dict
                Must contain 'time' (int/float) and 'dove_location' (float).
            """
            x = payload['dove_location']
            t = payload['time']
            self.add_to_quarantine(t, x)
            self.last_observed_data.append(x)
            self.current_x = x
            prev_x = self.pop_from_quarantine(t)

            if prev_x is not None:
                self.x_y_data.append((prev_x, x))

                if self.count > self.window_size and self.count % self.train_model_frequency == 0:

                    x_y_data = np.array(self.x_y_data)
                    xi_values = x_y_data[:, 0]
                    yi_values = x_y_data[:, 1]

                    # Determine the number of data points to use for training
                    num_data_points = min(len(xi_values), self.num_data_points_max)
                    if len(xi_values) < self.num_data_points_max + self.window_size:
                        num_data_points = num_data_points - (self.window_size + 3)

                    # Construct 'X' with fixed-size slices and 'y' as the values to predict
                    X = np.lib.stride_tricks.sliding_window_view(xi_values[-(num_data_points + self.window_size - 1):],
                                                                 self.window_size)
                    y = yi_values[-num_data_points:]

                    # Fit a single NGBoost model (since we only need one model)
                    self.model.fit(X, y)

                    # Keep only latest data (to limit memory usage as it will be run on continuous live data)
                    self.x_y_data = self.x_y_data[-(self.num_data_points_max + self.window_size * 2):]
                    self.last_observed_data = self.last_observed_data[-(self.window_size + 1):]
                self.count += 1

        def predict(self):
            """
            Return a dictionary representing the best guess of the distribution,
            modeled as a Gaussian distribution.
            """
            # the central value (mean) of the gaussian distribution will be represented by the current value
            x_mean = self.current_x
            components = []

            try:
                X_input = np.array([self.last_observed_data[-(self.window_size + 1):]])

                # Get the predicted distribution
                y_test_ngb = self.model.pred_dist(X_input)

                # here we use current value as loc but you can get the parameter loc from ngboost normal distribution class: y_test_ngb.loc[0]
                loc = x_mean

                scale = y_test_ngb.scale[0]  # get the parameter scale from ngboost normal distribution class
                scale = max(scale, 1e-6)
            except:
                loc = x_mean
                scale = 1e-6

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
    NGBoostTracker = None




if __name__ == '__main__':
    tracker = NGBoostTracker()
    tracker.test_run(
        live=False, # Set to True to use live streaming data; set to False to use data from a CSV file
        step_print=2000 # How often to print scores
    )