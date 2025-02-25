import abc
import numpy as np

from birdgame.datasources.livedata import live_data_generator
from tqdm.auto import tqdm

from densitypdf import density_pdf

from birdgame.datasources.remotetestdata import remote_test_data_generator


class TrackerBase:
    """
    Base class that handles quarantining of data points before they are eligible for processing.
    """

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.quarantine = []
        self.count = 0

    def add_to_quarantine(self, time, value):
        """ Adds a new value to the quarantine list. """
        self.quarantine.append((time + self.horizon, value))

    def pop_from_quarantine(self, current_time):
        """ Returns the most recent valid data point from quarantine. """
        valid = [(j, (ti, xi)) for (j, (ti, xi)) in enumerate(self.quarantine) if ti <= current_time]
        if valid:
            prev_ndx, (ti, prev_x) = valid[-1]
            self.quarantine = self.quarantine[:prev_ndx]  # Trim the quarantine list
            return prev_x
        return None

    @abc.abstractmethod
    def tick(self, payload: dict):
        pass

    @abc.abstractmethod
    def predict(self) -> dict:
        pass

    def test_run(self, live=True, step_print=1000):
        from birdgame.model_benchmark.emwavartracker import EMWAVarTracker

        benchmark_tracker = EMWAVarTracker(horizon=self.horizon)
        my_run, bmark_run = TrackerEvaluator(self), TrackerEvaluator(benchmark_tracker)

        gen = live_data_generator() if live else remote_test_data_generator()
        try:
            for i, payload in enumerate(tqdm(gen)):

                my_run.tick_and_predict(payload)
                bmark_run.tick_and_predict(payload)

                if (i + 1) % step_print == 0:
                    print(f"My score: {my_run.score():.4f} VS Benchmark score: {bmark_run.score():.4f}")

            print(f"My score: {my_run.score():.4f} VS Benchmark score: {bmark_run.score():.4f}")
        except KeyboardInterrupt:
            print("Interrupted")


class TrackerEvaluator:
    def __init__(self, tracker: TrackerBase):
        self.tracker = tracker
        self.quarantine = []
        self.scores = []

    def tick_and_predict(self, payload: dict):
        self.tracker.tick(payload)
        prediction = self.tracker.predict()

        current_time = payload['time']
        self.quarantine.append({
            "time": current_time,
            "prediction": prediction
        })

        previous_valid_item = next((item for item in reversed(self.quarantine) if item["time"] < current_time - self.tracker.horizon), None)
        if not previous_valid_item:
            return  # nothing to score

        self.quarantine = [item for item in self.quarantine if item["time"] >= previous_valid_item["time"]]

        density = density_pdf(density_dict=previous_valid_item["prediction"], x=payload['dove_location'])
        self.scores.append(density)


    def score(self):
        if not self.scores:
            print("No scores to average")
            return 0.0

        return float(np.median(self.scores))