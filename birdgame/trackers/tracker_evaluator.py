import numpy as np

from densitypdf import density_pdf

from birdgame.trackers.trackerbase import Quarantine, TrackerBase


class TrackerEvaluator(Quarantine):
    def __init__(self, tracker: TrackerBase):
        self.tracker = tracker
        self.scores = []
        super().__init__(tracker.horizon)

    def tick_and_predict(self, payload: dict):
        self.tracker.tick(payload)
        prediction = self.tracker.predict()

        current_time = payload['time']
        self.add_to_quarantine(current_time, prediction)
        prev_prediction = self.pop_from_quarantine(current_time)
        if not prev_prediction:
            return

        density = density_pdf(density_dict=prev_prediction, x=payload['dove_location'])
        self.scores.append(density)

    def score(self):
        if not self.scores:
            print("No scores to average")
            return 0.0

        return float(np.median(self.scores))
