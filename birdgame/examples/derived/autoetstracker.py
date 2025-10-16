import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm.auto import tqdm
import numpy as np
import threading
import warnings

warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")

from birdgame.trackers.trackerbase import TrackerBase
from birdgame import HORIZON

class AutoETSConstants:
    MIN_SAMPLES = 5
    TRAIN_MODEL_FREQUENCY=2
    NUM_DATA_POINTS_MAX=20
    WARMUP_CUTOFF=0
    USE_THREADING=True # Set this to True for live data streams where each `tick()` and `predict()` call must complete within ~50 ms


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
            The prediction horizon in seconds (how far into the future predictions should be made).
        train_model_frequency : int
            The frequency at which the sktime model will be retrained based on the count of observations 
            ingested. This determines how often the model will be updated with new data.
        num_data_points_max : int
            The maximum number of data points to use for training the sktime model.
        warmup : int
            The number of ticks taken to warm up the model (wealth does not change during this period).
        use_threading : bool
            Whether to retrain the model asynchronously in a background thread.  
            /!/ Set this to True for live data streams where each `tick()`  
            and `predict()` call must complete within ~50 ms.  
            When enabled, retraining happens in parallel without blocking predictions.
        """

        def __init__(self, horizon=HORIZON):
            super().__init__(horizon)
            self.current_x = None
            self.last_observed_data = [] # Holds the last few observed data points
            self.prev_t = 0

            self.min_samples = AutoETSConstants.MIN_SAMPLES
            self.train_model_frequency = AutoETSConstants.TRAIN_MODEL_FREQUENCY
            self.num_data_points_max = AutoETSConstants.NUM_DATA_POINTS_MAX

            # Number of steps to predict
            steps = 1 # only one because the univariate serie will only have values separated of at least HORIZON time
            self.fh = np.arange(1, steps + 1)

            # Fit the AutoETS forecaster (no seasonality)
            self.forecaster = AutoETS(auto=True, sp=1, information_criterion="aic")
            self.scale = 1e-6

            # or Fit the AutoARIMA forecaster
            # self.forecaster = AutoARIMA(max_p=2, max_d=1, max_q=2, maxiter=10)

            self.warmup_cutoff = AutoETSConstants.WARMUP_CUTOFF
            self.tick_count = 0

            # Threading tools
            self.use_threading = AutoETSConstants.USE_THREADING
            self._lock = threading.Lock()
            if self.use_threading:
                self._cond = threading.Condition(self._lock)
                self._new_data = None
                self._stop_worker = False
                self._worker_thread = threading.Thread(target=self._worker_retrain_model_async, daemon=True)
                self._worker_thread.start()

        # ------------------- Tick -------------------
        def tick(self, payload, performance_metrics=None):
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
            
            x = payload["dove_location"]
            t = payload["time"]
            self.add_to_quarantine(t, x)
            self.current_x = x

            # Collect and process observations only at horizon-based intervals
            if t > self.prev_t + self.horizon:
                self.last_observed_data.append(x)
                self.prev_t = t

                if self.count == self.min_samples or (self.count > self.min_samples and self.count % self.train_model_frequency == 0):
                    # Construct 'y' as an univariate serie
                    y = np.array(self.last_observed_data)[-self.num_data_points_max:]

                    # Fit sktime model and variance prediction
                    if self.use_threading:
                        # Signal background thread
                        with self._cond:
                            self._new_data = y
                            self._cond.notify()
                    else:
                        self._retrain_model_sync(y)

                    # Update last observed data (to limit memory usage as it will be run on continuous live data)
                    self.last_observed_data = self.last_observed_data[-(self.num_data_points_max + 2):]

                self.count += 1

            self.tick_count += 1

        # ------------------- Prediction -------------------
        def predict(self):
            """
            Return a dictionary representing the best guess of the distribution,
            modeled as a Gaussian distribution.

            If the model is in the warmup period, return None.
            """
            with self._lock:
                # Check if the model is warming up
                if self.tick_count < self.warmup_cutoff or self.forecaster is None:
                    return None

                # the central value (mean) of the gaussian distribution will be represented by the current value
                # but you can get point forecast from 'self.forecaster.predict(fh=self.fh[-1])[0][0]'
                loc = self.current_x
                # we predicted scale during tick training
                scale = max(getattr(self, "scale", 1e-6), 1e-6)

            # time.sleep(0.01)  # mimic short inference delay

            # Return the prediction density
            components = {
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {"loc": loc, "scale": scale},
                },
                "weight": 1,
            }

            return {"type": "mixture", "components": [components]}

        # ------------------- Model training -------------------
        def _fit(self, y):
            # Fit a clone sktime model (at least a cloned model is required in case of asynchronous training)
            new_forecaster = self.forecaster.clone()
            new_forecaster.fit(y, fh=self.fh)
            # Variance prediction
            var = new_forecaster.predict_var(fh=self.fh)
            scale = np.sqrt(var.values.flatten()[-1])

            return new_forecaster, scale

        def _retrain_model_sync(self, y):
            """Synchronous retraining"""
            start_time = time.perf_counter()
            self.forecaster, self.scale = self._fit(y)
            # print(f"Sync retrain time: {(time.perf_counter()- start_time)*1000:.2f} ms") # check training time

        def _worker_retrain_model_async(self):
            """Asynchronous retraining in a background worker"""
            while True:
                with self._cond:
                    # Wait until new data is available
                    while self._new_data is None:
                        self._cond.wait()
                    y = self._new_data  # get the data to train on
                    self._new_data = None  # clear it (so next signal is new data)

                # Train the model outside the lock (so predict() can still run)
                new_forecaster, scale = self._fit(y)

                # Swap the trained model safely
                with self._lock:
                    self.forecaster = new_forecaster
                    self.scale = scale
                print("Async retraining done")

else:
    AutoETSsktimeTracker = None



if __name__ == '__main__':
    live=False # Set to True to use live streaming data; set to False to use data from a CSV file
    AutoETSConstants.USE_THREADING = live
    tracker = AutoETSsktimeTracker()
    tracker.test_run(
        live=live,
        step_print=10000 # Print the score and progress every 1000 steps
    )