import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
from birdgame.trackers.trackerbase import TrackerBase
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")

class AutoETSConstants:
    HORIZON = 10
    TRAIN_MODEL_FREQUENCY=50
    NUM_DATA_POINTS_MAX=20


try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from sktime.forecasting.ets import AutoETS
    from sktime.forecasting.base import ForecastingHorizon
    import warnings
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    from sktime.forecasting.arima import AutoARIMA
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    using_sktime = True
except ImportError:
    print('To run this example you need to pip install statsmodels sktime')
    using_sktime = False

if using_sktime:

    class AutoETSsktimeTracker(TrackerBase):
        """
        A model that tracks the dove location using AutoETS.

        Parameters
        ----------
        horizon : int
            The number of time steps into the future that predictions should be made for.
        train_model_frequency : int
            The frequency at which the sktime model will be retrained based on the count of observations
            ingested. This determines how often the model will be updated with new data.
        num_data_points_max : int
            The maximum number of data points to use for training the sktime model.
        """

        def __init__(self):
            super().__init__(AutoETSConstants.HORIZON)
            self.current_x = None
            self.last_observed_data = []  # Holds the last few observed data points
            self.prev_t = 0

            self.train_model_frequency = AutoETSConstants.TRAIN_MODEL_FREQUENCY
            self.num_data_points_max = AutoETSConstants.NUM_DATA_POINTS_MAX

            # Number of steps to predict
            steps = 1  # only one because the univariate serie will only have values separated of at least HORIZON time
            self.fh = np.arange(1, steps + 1)

            # Fit the AutoETS forecaster (no seasonality)
            self.forecaster = AutoETS(auto=True, sp=1, information_criterion="aic")

            # or Fit the AutoARIMA forecaster
            # self.forecaster = AutoARIMA(max_p=2, max_d=1, max_q=2, maxiter=10)

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
            self.current_x = x

            # we build a univariate serie of values separated of at least HORIZON time
            if t > self.prev_t + self.horizon:
                self.last_observed_data.append(x)
                self.prev_t = t

            prev_x = self.pop_from_quarantine(t)

            if prev_x is not None:

                if self.count > 10 and self.count % self.train_model_frequency == 0:
                    # Construct 'y' as an univariate serie
                    y = np.array(self.last_observed_data)[-self.num_data_points_max:]

                    # Fit sktime model
                    self.forecaster.fit(y, fh=self.fh)

                    # Variance prediction
                    var = self.forecaster.predict_var(fh=self.fh)
                    self.scale = np.sqrt(var.values.flatten()[-1])

                    # Update last observed data (to limit memory usage as it will be run on continuous live data)
                    self.last_observed_data = self.last_observed_data[-(self.num_data_points_max + 2):]
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
                # here we use current value as loc but you can get point forecast from 'self.forecaster.predict(fh=self.fh[-1])[0][0]'
                loc = x_mean

                # we predicted scale during tick training
                scale = self.scale
                scale = max(scale, 1e-6)

                # If you want to predict variance for each prediction
                # scale = self.forecaster.predict_var(fh=self.fh)
                # scale = np.sqrt(scale.values.flatten()[-1])
            except:
                loc = x_mean
                scale = 1e-6

            # Return the prediction density
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
    AutoETSsktimeTracker = None



if __name__ == '__main__':
    tracker = AutoETSsktimeTracker()
    tracker.test_run(
        live=False, # Set to True to use live streaming data; set to False to use data from a CSV file
        step_print=10000 # Print the score and progress every 1000 steps
    )