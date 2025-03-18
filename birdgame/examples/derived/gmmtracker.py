from pprint import pprint
import math
import numpy as np
from birdgame.trackers.trackerbase import TrackerBase
from birdgame.datasources.livedata import live_data_generator
from densitypdf import density_pdf
from birdgame.examples.derived.mixturetracker import MixtureTracker

# scikit-learn GMM
try:
    from sklearn.mixture import GaussianMixture
    using_sklearn=True
except ImportError:
    using_sklearn = False

if using_sklearn:


    class GMMTracker(TrackerBase):
        """
        A tracker that runs two models simultaneously:
          1) MixtureTracker (the fallback)
          2) A scikit-learn GaussianMixture-based model that:
             - collects horizon-based differences
             - shifts the data mean by 'data_shrinkage' * sample_mean before fitting,
               so that the final variance is unchanged but the data's center is pulled toward 0.

        We keep using MixtureTracker's distribution in predict() until 'burn_in' observations.
        After burn_in, we switch to the GMM's distribution.
        """

        def __init__(
            self,
            n_components=2,
            horizon=3,
            batch_size=500,
            burn_in=2000,
            data_shrinkage=0.0,
            window_len=10000
        ):
            """
            Args:
                n_components (int): Number of Gaussian components in the GMM
                horizon (int): # steps in quarantine before we pop from it
                batch_size (int): # of differences to accumulate before re-fitting GMM
                burn_in (int): # observations until we switch to GMM predictions
                data_shrinkage (float): fraction of the sample mean to remove from X before fitting
            """
            super().__init__(horizon)

            # 1) Fallback tracker
            self.fallback = MixtureTracker(horizon=horizon)
            self.window_len = window_len

            # 2) GMM-based approach
            self.gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                random_state=42,
                max_iter=200
            )
            self.is_fitted = False
            self.x_changes = []
            self.batch_size = batch_size
            self.data_shrinkage = data_shrinkage

            # Misc
            self.count = 0
            self.current_x = None
            self.burn_in = burn_in

        def tick(self, payload):
            """
            Tick both the fallback MixtureTracker and also gather data for the GMM model.
            """
            # 1) Always update fallback
            self.fallback.tick(payload)

            # 2) GMM horizon logic
            x = payload["dove_location"]
            t = payload["time"]
            self.add_to_quarantine(t, x)
            prev_x = self.pop_from_quarantine(t)
            self.current_x = x
            self.count += 1

            # Once we pop from quarantine, we can form a difference
            if prev_x is not None:
                x_change = x - prev_x
                self.x_changes.append(x_change)

                # Fit GMM if we have enough new differences
                if len(self.x_changes) % self.batch_size==0:
                    self._refit_gmm()
                    self.x_changes = self.x_changes[-self.window_len:]

        def _refit_gmm(self):
            """
            Shift the data's mean by data_shrinkage * sample_mean => no variance scaling.
            Then fit the GMM.
            """
            X = np.array(self.x_changes, dtype=np.float32).reshape(-1, 1)
            if X.shape[0] < 2:
                return

            # 1) Find sample mean
            mean_x = X.mean()  # shape: ()
            # 2) Shift = data_shrinkage * mean_x
            shift_value = self.data_shrinkage * mean_x
            # 3) X_shifted => X - shift_value
            X_shifted = X - shift_value

            self.gmm.fit(X_shifted)
            self.is_fitted = True
            # Store the shift_value used for unshifting in predict
            self.latest_shift_value = float(shift_value)

        def predict(self):
            """
            If we've done fewer than 'burn_in' ticks (or the GMM isn't fitted),
            return fallback's distribution. Otherwise, return GMM distribution.
            """
            # If not enough observations or no GMM fit yet, fallback
            if (not self.is_fitted) or (self.count < self.burn_in):
                return self.fallback.predict()
            else:
                return self.gmm_predict()

        def gmm_predict(self):
            """
            Return the GMM distribution in the original coordinate system.
            Means are unshifted by 'latest_shift_value', then we add current_x
            to shift from difference domain -> absolute positions.
            """
            if not self.is_fitted:
                # Something unexpected. Fallback.
                return self.fallback.predict()

            means = self.gmm.means_.ravel()
            covars = self.gmm.covariances_
            weights = self.gmm.weights_

            shift_val = getattr(self, "latest_shift_value", 0.0)
            mu_x = self.current_x if self.current_x is not None else 0.0

            components = []
            for k, w_k in enumerate(weights):
                # GMM means are in the shifted domain => unshift
                unshifted_mean = means[k] + shift_val

                # No scaling on var => unchanged
                var_k = covars[k][0, 0]
                std_k = math.sqrt(var_k) if var_k > 0 else 1e-6

                # Then from difference domain to absolute => + current_x
                final_mean = float(mu_x + unshifted_mean)

                components.append({
                    "density": {
                        "type": "builtin",
                        "name": "norm",
                        "params": {"loc": final_mean, "scale": float(std_k)}
                    },
                    "weight": float(w_k)
                })

            mixture_dict = {
                "type": "mixture",
                "components": components
            }
            _ = density_pdf(mixture_dict, x=0.0)
            return mixture_dict
else:
    GMMTracker = None 



if __name__ == '__main__':
    tracker = GMMTracker(
        n_components=2,
        horizon=3,
        batch_size=200,
        burn_in=1000,
        window_len=10000,
        data_shrinkage=0.95  # remove 'data_shrinkage'  of the data's mean before fitting
    )
    tracker.test_run(live=True, step_print=200)
