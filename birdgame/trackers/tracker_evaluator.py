import numpy as np

from densitypdf import density_pdf

from birdgame.trackers.trackerbase import Quarantine, TrackerBase


class TrackerEvaluator(Quarantine):
    def __init__(self, tracker: TrackerBase):
        self.tracker = tracker
        self.scores = []
        super().__init__(tracker.horizon)

        self.time = None
        self.loc = None
        self.scale = None
        self.dove_location = None

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

        self.time = current_time
        self.dove_location = payload['dove_location']
        self.update_loc_and_scale(density_dict=prev_prediction)

    def score(self):
        if not self.scores:
            print("No scores to average")
            return 0.0

        return float(np.median(self.scores))
    
    def update_loc_and_scale(self, density_dict):
        dist_type = density_dict.get("type")
        if dist_type == "mixture":
            # if mixture: get loc and scale from highest weight distribution
            # Get index of the highest weight
            max_index = max(range(len(density_dict["components"])), key=lambda i: density_dict["components"][i]["weight"])
            density_dict = density_dict["components"][max_index]["density"]
        params = density_dict["params"]
        self.loc = params.get("loc", params.get("mu", None))
        self.scale = params.get("scale", params.get("sigma", None))
