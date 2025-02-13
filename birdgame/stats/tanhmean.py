import math
import numpy as np

from birdgame.stats.fewmean import FEWMean
from birdgame.stats.fewvar import FEWVar
from jumpdiffusion import jump_diffusion


# Just messin' around use at your peril

def tanh_scale(x, alpha=2.0):
    """

        Intended to relate the outlier fraction to the scale of the tanh function.

    :param x:
    :param alpha:
    :return:
    """
    return 2*alpha ** (x / alpha)


class TanhMean:
    """
    Sublinear FEWMean estimator with reaction to running mean of outlier indicator.
    Uses parameters:
      - fading_factor
      - outlier_fading_factor
      - alpha
    """

    def __init__(self,
                 mean_fading_factor=0.1,
                 var_fading_factor=0.01,
                 outlier_fading_factor=0.3,
                 alpha=0.2):

        # Params
        self.mean_fading_factor = mean_fading_factor
        self.var_fading_factor = var_fading_factor
        self.outlier_fading_factor = outlier_fading_factor
        self.alpha = alpha

        # State
        self._mean = FEWVar(fading_factor=self.mean_fading_factor)
        self._var = FEWVar(fading_factor=self.var_fading_factor)
        self._outlier_ewa = FEWMean(fading_factor=self.outlier_fading_factor)
        for _ in range(10):
            self._outlier_ewa.update(0)
        self.n = 0

    def get_mean(self):
        return self._mean.get_mean()

    def get_var(self):
        return self._var.get_var()

    def update(self, x):
        self.n += 1
        x_mean = self._mean.get_mean()
        x_var = self._var.get_var()

        if x_var is None or x_var == 0:
            x_synthetic = x
        else:
            x_std = math.sqrt(x_var)
            z_score = (x - x_mean) / x_std
            outlier_ternary_indicator = int(abs(z_score) > 0.5) * np.sign(z_score)
            self._outlier_ewa.update(outlier_ternary_indicator)
            outlier_mean = self._outlier_ewa.get_mean()
            scale = tanh_scale(outlier_mean * np.sign(z_score), alpha=self.alpha)
            z_ratio = math.tanh(abs(z_score) / scale) / math.tanh(1 / scale)
            x_synthetic = x_mean + z_ratio * (x - x_mean)

            # Catch up
            if outlier_mean*np.sign(z_score)>0.25:
                self._mean.update(x_synthetic)
            if outlier_mean*np.sign(z_score)>0.4:
                self._mean.update(x_synthetic)
            if outlier_mean * np.sign(z_score)>0.6:
                self._mean.update(x_synthetic)

        self._mean.update(x_synthetic)
        self._var.update(x_synthetic-x_mean)

    def apply_series(self, series, burn_in=100):
        """
        Apply the estimator to a given series and return a performance metric.

        This is a placeholder implementation.
        It computes residuals after burn-in and returns the sum of squared residuals as a metric.
        You should adjust it to suit your actual evaluation criterion.
        """
        # Reset internal state
        self.__init__(
            mean_fading_factor=self.mean_fading_factor,
            outlier_fading_factor=self.outlier_fading_factor,
            alpha=self.alpha
        )

        residuals = []
        for i, x in enumerate(series):
            self.update(x)
            if i >= burn_in:
                residual = self.get_mean() - x
                residuals.append(residual)

        # Example metric: sum of squared residuals
        return sum(r ** 2 for r in residuals) if residuals else float('inf')

    def get_params(self):
        """
        Return current parameters of the estimator.
        """
        return {
            "fading_factor": self.mean_fading_factor,
            "outlier_fading_factor": self.outlier_fading_factor,
            "alpha": self.alpha
        }

    def fit_to_simulation(self,
                          n_sim=10000,
                          jump_rate=0.01,
                          jump_size=1.0,
                          param_grids=None,
                          epsilon=0.3,
                          vega=1.0):
        """
        Fit parameters by simulating data and optimizing a performance metric.

        Parameters
        ----------
        n_sim : int
            Number of simulated steps.
        jump_rate : float
            Probability of a jump at any given step.
        jump_size : float
            Magnitude of jumps.
        param_grids : dict or None
            Dictionary specifying lists of values to try for each parameter.
        epsilon : float
            Noise scale for the measurement noise.
        vega : float
            Noise scale for exponential measurement error.

        Returns
        -------
        best_params : dict
            Dictionary of parameters that yield the best metric.
        """

        if param_grids is None:
            param_grids = {
                "alpha": [0.1, 0.2, 0.3],
                "outlier_fading_factor": [0.1, 0.2, 0.3, 0.4, 0.5],
                "mean_fading_factor": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5],
                "var_fading_factor": [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
            }

        # Simulate the series
        series = jump_diffusion(n_sim, jump_rate=jump_rate, jump_size=jump_size, epsilon=epsilon, vega=vega)

        best_metric = float('inf')
        best_params = self.get_params()

        from itertools import product
        keys = list(param_grids.keys())
        value_combinations = product(*param_grids.values())

        for values in value_combinations:
            trial_params = {k: v for k, v in zip(keys, values)}

            # Temporarily set these parameters
            self.alpha = trial_params["alpha"]
            self.outlier_fading_factor = trial_params["outlier_fading_factor"]
            self.mean_fading_factor = trial_params["mean_fading_factor"]
            self.var_fading_factor = trial_params["var_fading_factor"]

            # Re-initialize internal state as needed
            self.__init__(
                mean_fading_factor=self.mean_fading_factor,
                outlier_fading_factor=self.outlier_fading_factor,
                alpha=self.alpha
            )

            metric = self.apply_series(series)

            if metric < best_metric:
                best_metric = metric
                best_params = {
                    "mean_fading_factor": self.mean_fading_factor,
                    "var_fading_factor": self.var_fading_factor,
                    "outlier_fading_factor": self.outlier_fading_factor,
                    "alpha": self.alpha
                }

        # Restore best found params
        self.__init__(**best_params)

        return best_params




if __name__ == '__main__':
    from pprint import pprint
    tm = TanhMean(mean_fading_factor=0.05)
    best_params = tm.fit_to_simulation(n_sim=10000, jump_rate=0.01, jump_size=20, epsilon=0.3, vega=2.0)
    pprint(best_params)
    r_best = TanhMean(**best_params)