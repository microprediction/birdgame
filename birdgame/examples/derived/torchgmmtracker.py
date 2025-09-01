from birdgame.trackers.trackerbase import TrackerBase
import threading
import torch
import numpy as np

try:
    from tgmm import GaussianMixture
    using_tgmm = True
except ImportError:
    using_tgmm = False

class TorchGMMTracker(TrackerBase):
    """
    A tracker that computes (x - prev_x) using the same 'quarantine'
    horizon logic as ConformalEMWAVarTracker, then batches those
    differences and refits a 2-component Gaussian mixture model in a
    background thread every N differences.

    - Reads 'dove_location' and 'time' from payload
    - Uses horizon-based "quarantine" to get x_change
    - Accumulates x_change in a list
    - Spawns a background thread to refit the GMM every 'batch_size' points
    - Lock-protects GMM during refit and prediction
    """

    def __init__(
        self,
        horizon=3,
        n_components=2,
        batch_size=1000
    ):
        """
        Args:
            horizon (int): how many ticks before we pop data from quarantine
            n_components (int): number of Gaussians in the mixture
            batch_size (int): how many data points to buffer before refitting
        """
        super().__init__(horizon)

        self.batch_size = batch_size
        self.differences = []

        # Only create the model if tgmm is available
        self.using_tgmm = using_tgmm
        if self.using_tgmm:
            # Initialize a GMM for 1D data
            self.gmm = GaussianMixture(
                n_features=1,        # 1D differences
                n_components=n_components,
                covariance_type="full",
                max_iter=50,
                init_params="kmeans",
                device="cpu"
            )
        else:
            self.gmm = None

        # Concurrency
        self.model_lock = threading.Lock()
        self.fit_thread = None

        # For logging or reference
        self.count = 0

    def tick(self, payload, performance_metrics):
        """
        1) Parse x, t from the payload
        2) add_to_quarantine(t, x)
        3) pop_from_quarantine(t) => prev_x
        4) If prev_x is not None, compute (x - prev_x) => store in self.differences
        5) If self.differences has >= batch_size items, spawn a thread to refit GMM
        """
        if not self.using_tgmm:
            # If tgmm not installed, just return or do some fallback
            return

        self.count += 1

        # 1) read from payload
        x = payload["dove_location"]
        t = payload["time"]

        # 2) quarantine logic
        self.add_to_quarantine(t, x)
        prev_x = self.pop_from_quarantine(t)

        # 3) if we have a previous location, store the difference
        if prev_x is not None:
            x_change = x - prev_x
            self.differences.append(x_change)

        # 4) if we exceed batch_size, refit in the background
        if len(self.differences) >= self.batch_size:
            data_array = np.array(self.differences, dtype=np.float32)
            self.differences.clear()

            # Check if a previous thread is running
            if self.fit_thread is not None and self.fit_thread.is_alive():
                # If a refit is still in progress, we can choose to skip or queue
                # We'll just skip for brevity
                return

            # Spawn background refit
            self.fit_thread = threading.Thread(
                target=self._refit_model, args=(data_array,)
            )
            self.fit_thread.start()

    def _refit_model(self, data_array):
        """Runs in a background thread. Locks, converts to Torch, fits GMM, unlocks."""
        with self.model_lock:
            if self.gmm is None:
                return
            X = torch.from_numpy(data_array)  # shape: (batch_size, 1)
            self.gmm.fit(X)

    def predict(self):
        """
        Return the GMM mixture distribution as a dictionary with two components,
        matching the same schema as ConformalEMWAVarTracker's predict().
        If GMM is uninitialized or tgmm not installed, return a default mixture.
        """
        # If GMM not available or hasn't been fit yet, provide fallback
        if (not self.using_tgmm) or (self.gmm is None):
            return {
                "type": "mixture",
                "components": [
                    {
                        "density": {
                            "type": "builtin",
                            "name": "norm",
                            "params": {"loc": 0.0, "scale": 1.0}
                        },
                        "weight": 1.0
                    }
                ]
            }

        with self.model_lock:
            means = self.gmm.means_.detach().cpu().numpy()        # shape: (n_components, 1)
            covars = self.gmm.covariances_.detach().cpu().numpy() # shape: ...
            weights = self.gmm.weights_.detach().cpu().numpy()    # shape: (n_components,)

        mixture = {"type": "mixture", "components": []}
        for i, w in enumerate(weights):
            mu = float(means[i][0])   # 1D mean
            # If 'full' covariance for 1D => a 1x1 matrix => just take sqrt
            sigma = float(np.sqrt(covars[i][0]))
            mixture["components"].append({
                "density": {
                    "type": "builtin",
                    "name": "norm",
                    "params": {"loc": mu, "scale": sigma}
                },
                "weight": float(w)
            })
        return mixture


if __name__ == "__main__":
    # Simple demonstration (only works if tgmm is installed)
    tracker = TorchGMMTracker(horizon=3, n_components=2, batch_size=1000)
    tracker.test_run(live=True,step_print=100)
