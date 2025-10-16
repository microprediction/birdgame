import os
import math
import time
import pandas as pd
import numpy as np
from birdgame.trackers.trackerbase import TrackerBase
from birdgame import HORIZON
import threading
import warnings

warnings.filterwarnings("ignore", message="Non-stationary starting autoregressive parameters")
warnings.filterwarnings("ignore", message="Non-invertible starting MA parameters found")


class NGBoostConstants:
    TRAIN_MODEL_FREQUENCY=100
    NUM_DATA_POINTS_MAX=1000
    WINDOW_SIZE = 5
    WARMUP_CUTOFF = 0
    USE_THREADING=True # Set this to True for live data streams where each `tick()` and `predict()` call must complete within ~50 ms

try:
    from ngboost import NGBoost
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.base import clone
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
        A model that tracks the dove location using NGBoost.

        Parameters
        ----------
        horizon : int
            The prediction horizon in seconds (how far into the future predictions should be made).
        train_model_frequency : int
            The frequency at which the NGBoost model will be retrained based on the count of observations 
            ingested. This determines how often the model will be updated with new data.
        num_data_points_max : int
            The maximum number of data points to use for training the NGBoost model.
        window_size : int
            The number of previous data points (the sliding window size) used to predict the future value 
            at the horizon. It defines how many past observations are considered for prediction.
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
            self.x_y_data = [] # Holds pairs of previous and current data points

            self.train_model_frequency = NGBoostConstants.TRAIN_MODEL_FREQUENCY
            self.num_data_points_max = NGBoostConstants.NUM_DATA_POINTS_MAX # (X.shape[0])
            self.window_size = NGBoostConstants.WINDOW_SIZE # (X.shape[1])
            self.warmup_cutoff = NGBoostConstants.WARMUP_CUTOFF
            self.use_threading = NGBoostConstants.USE_THREADING

            # Initialize NGBoost model
            self.model = NGBoost(
                Dist=Normal,
                learning_rate=0.1,
                n_estimators=50,
                natural_gradient=True,
                verbose=False,
                random_state=15,
                validation_fraction=0.1,
                early_stopping_rounds=None,
                Base=DecisionTreeRegressor(
                    criterion="friedman_mse",
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_depth=5,
                    splitter="best",
                ),
            )

            # Internal counters
            self.tick_count = 0

            # Threading tools
            self._lock = threading.Lock()
            if self.use_threading:
                self._lock = threading.Lock()
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
            self.last_observed_data.append(x)
            self.current_x = x
            prev_x = self.pop_from_quarantine(t)

            if prev_x is not None:
                self.x_y_data.append((prev_x, x))

                # retraining condition
                if self.count > self.window_size and self.count % self.train_model_frequency == 0:
                    x_y_data = np.array(self.x_y_data)
                    xi_values = x_y_data[:, 0]
                    yi_values = x_y_data[:, 1]

                    # Determine the number of data points to use for training
                    num_data_points = min(len(xi_values), self.num_data_points_max)
                    if len(xi_values) < self.num_data_points_max + self.window_size:
                        num_data_points = max(0, num_data_points - (self.window_size + 3))

                    if num_data_points > self.window_size + 2:
                        # Construct 'X' with fixed-size slices and 'y' as the values to predict
                        X = np.lib.stride_tricks.sliding_window_view(
                            xi_values[-(num_data_points + self.window_size - 1):],
                            self.window_size,
                        )
                        y = yi_values[-num_data_points:]

                        # Fit a single NGBoost model (since we only need one model)
                        if self.use_threading:
                            with self._cond:
                                self._new_data = (X, y)  # overwrite old requests
                                self._cond.notify()
                        else:
                            self._retrain_model_sync(X, y)

                    # Keep only latest data (to limit memory usage as it will be run on continuous live data)
                    self.x_y_data = self.x_y_data[-(self.num_data_points_max + self.window_size * 2):]
                    self.last_observed_data = self.last_observed_data[-(self.window_size + 1):]

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
                if self.tick_count < self.warmup_cutoff:
                    return None
                
                start_time = time.perf_counter()

                # the central value (mean) of the gaussian distribution will be represented by the current value
                x_mean = self.current_x
                try:
                    X_input = np.array([self.last_observed_data[-self.window_size:]])
                    y_test_ngb = self.model.pred_dist(X_input)
                    loc = x_mean  # can use y_test_ngb.loc[0] if you prefer model mean
                    scale = max(y_test_ngb.scale[0], 1e-6) # get the parameter scale from ngboost normal distribution class
                except Exception:
                    loc = x_mean
                    scale = 1e-6

                elapsed_ms = (time.perf_counter() - start_time) * 1000  # ms
                # print(f"predict() took {elapsed_ms:.2f} ms")
                if elapsed_ms > 50:
                    print(f"predict() took {elapsed_ms:.2f} ms")

            # time.sleep(0.01)  # mimic short inference delay

            components = {
                "density": {"type": "builtin", "name": "norm", "params": {"loc": loc, "scale": scale}},
                "weight": 1,
            }
            return {"type": "mixture", "components": [components]}

        # ------------------- Model training -------------------
        def _fit(self, X, y):
            """Train a fresh NGBoost model and return it."""
            if X.shape[0] < 5:
                return self.model  # skip tiny samples

            new_model = clone(self.model)
            new_model.fit(X, y)
            return new_model

        def _retrain_model_sync(self, X, y):
            """Synchronous retraining."""
            start_time = time.perf_counter()
            self.model = self._fit(X, y)
            # print(f"Sync retrain time: {(time.perf_counter()- start_time)*1000:.2f} ms") # check training time

        def _worker_retrain_model_async(self):
            """Asynchronous retraining in a background worker"""
            while True:
                with self._cond:
                    # Wait until new data is available
                    while self._new_data is None:
                        self._cond.wait()
                    X, y = self._new_data  # get the data to train on
                    self._new_data = None  # clear it (so next signal is new data)

                # Train the model outside the lock (so predict() can still run)
                new_model = self._fit(X, y)

                # Swap the trained model safely
                with self._lock:
                    self.model = new_model
                # print("Async retraining done")

else:
    NGBoostTracker = None




if __name__ == '__main__':
    live=True # Set to True to use live streaming data; set to False to use data from a CSV file
    NGBoostConstants.USE_THREADING = live    
    tracker = NGBoostTracker()
    tracker.test_run(
        live=live,
        step_print=2000 # How often to print scores
    )