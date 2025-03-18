import numpy as np
from collections import deque

from densitypdf import density_pdf

from birdgame.trackers.trackerbase import Quarantine, TrackerBase


def robust_mean_log_like(scores):
    log_scores = np.log(1e-10 + np.array(scores))
    return np.mean(log_scores)


class TrackerEvaluator(Quarantine):
    def __init__(self, tracker: TrackerBase, score_window_size: int = 100):
        """
        Evaluates a given tracker by comparing its predictions to the actual dove locations.

        Parameters
        ----------
        tracker : TrackerBase
            The tracker instance to evaluate.
        score_window_size : int, optional
            The number of most recent scores to retain for computing the median latest score.
        """
        
        super().__init__(tracker.horizon)
        self.tracker = tracker
        self.scores = []
        self.score_window_size = score_window_size
        self.latest_scores = deque(maxlen=score_window_size)  # Keeps only the last `score_window_size` scores

        self.time = None
        self.dove_location = None
        self.latest_valid_prediction = None

    def tick_and_predict(self, payload: dict):
        """
        Process a new data point, make a prediction and evaluate it.
        """
        self.tracker.tick(payload)
        prediction = self.tracker.predict()

        current_time = payload['time']
        self.add_to_quarantine(current_time, prediction)
        prev_prediction = self.pop_from_quarantine(current_time)

        if not prev_prediction:
            return

        density = density_pdf(density_dict=prev_prediction, x=payload['dove_location'])
        self.scores.append(density)
        self.latest_scores.append(density) # Maintain a rolling window of recent scores
        self.latest_valid_prediction = prev_prediction

        self.time = current_time
        self.dove_location = payload['dove_location']

    def overall_likelihood_score(self):
        """
        Return the mean log-likelihood score over all recorded scores.
        """
        if not self.scores:
            print("No scores to average")
            return 0.0

        return float(robust_mean_log_like(self.scores))
    
    def recent_likelihood_score(self):
        """
        Return the mean log-likelihood score of the most recent `score_window_size` scores.
        """
        if not self.latest_scores:
            print("No recent scores available.")
            return 0.0
        
        return float(robust_mean_log_like(self.latest_scores))