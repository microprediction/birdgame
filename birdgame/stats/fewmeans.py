import math
from collections import deque
from birdgame.stats.fewmean import FEWMean
from birdgame.stats.fewvar import FEWVar


class FEWMeans:
    """
    Tracks running means using multiple different forgetting factors.
    After each update, it determines which two have been the most accurate
    over the last five observations and returns a weighted linear combination
    of those two means as the final estimate.

    Methods:
        __init__(fading_factors: list[float]):
            Initialize a FEWMeans with a list of fading factors.

        update(x: float):
            Incorporate a new data point x. Before updating, measure the prediction
            error of each mean. After updating, recompute which means are the best
            and store a combined estimate.

        get():
            Returns the current best combined estimate.

    Internal logic:
    - Each FEWMean tracks a running mean.
    - FEWMeans stores a rolling buffer of errors (up to last 5) for each FEWMean.
    - After update, it selects the two FEWMeans with the lowest average error over
      the last 5 observations and computes a weighted combination:
        weight_i = 1/(avg_error_i + epsilon)
      The final estimate = sum(mean_i * weight_i) / sum(weight_i) for the top two.
    """

    def __init__(self, fading_factors, var_fading_factor=0.01, buffer_size=5, epsilon=1e-9):
        self.fading_factors = fading_factors
        self.means = [FEWMean(f) for f in fading_factors]
        self.var = FEWVar(fading_factor=var_fading_factor)
        self.errors = [deque(maxlen=buffer_size) for _ in self.means]
        self.epsilon = epsilon
        self.current_estimate = None

    def update(self, x):
        if self.current_estimate is not None:
            self.var.update(x-self.current_estimate)

        # Get predictions before update
        predictions = [m.get() for m in self.means]

        # Compute and store errors
        for i, pred in enumerate(predictions):
            err = abs(x - pred)
            self.errors[i].append(err)

        # Update each FEWMean
        for m in self.means:
            m.update(x)

        # Determine which two FEWMeans are best
        # Compute average error over last 5 observations for each mean
        avg_errors = []
        for i, err_buffer in enumerate(self.errors):
            if len(err_buffer) > 0:
                avg_errors.append((i, sum(err_buffer)/len(err_buffer)))
            else:
                # If no errors yet, treat as perfect (or neutral)
                avg_errors.append((i, 0))

        # Sort by average error ascending (best = smallest error)
        avg_errors.sort(key=lambda x: x[1])

        # Take the top two means (or one if only one is available)
        top_means = avg_errors[:2]

        # If we have only one mean, just use that mean's value directly
        if len(top_means) == 1:
            best_idx = top_means[0][0]
            self.current_estimate = self.means[best_idx].get()
            return

        # Otherwise, compute weighted combination
        # weight_i = 1/(avg_error_i + epsilon)
        weights = []
        means_values = []
        for idx, aerr in top_means:
            w = 1.0 / (aerr + self.epsilon)
            weights.append(w)
            means_values.append(self.means[idx].get())

        total_weight = sum(weights)
        weighted_sum = sum(mv * w for mv, w in zip(means_values, weights))
        self.current_estimate = weighted_sum / total_weight


    def get(self):
        # Return the current combined estimate
        return self.current_estimate if self.current_estimate is not None else 0.0

    def get_mean(self):
        return self.get()

    def get_var(self):
        return self.var.get_var()

    def to_dict(self):
        """
        Serializes the state of the FEWMeans object and all internal FEWMean objects to a dictionary.
        """
        return {
            'fading_factors': self.fading_factors,
            'means': [m.to_dict() for m in self.means],
            'errors': [list(e) for e in self.errors],
            'current_estimate': self.current_estimate
        }

    @classmethod
    def from_dict(cls, data):
        """
        Deserializes the state from a dictionary into a new FEWMeans instance.
        """
        instance = cls(fading_factors=data['fading_factors'])
        instance.current_estimate = data['current_estimate']
        # Restore each FEWMean
        instance.means = [FEWMean.from_dict(mdict) for mdict in data['means']]
        # Restore errors
        instance.errors = [deque(e, maxlen=5) for e in data['errors']]
        return instance