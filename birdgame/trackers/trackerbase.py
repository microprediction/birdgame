import abc

from birdgame.datasources.livedata import live_data_generator
from tqdm.auto import tqdm

from birdgame.datasources.remotetestdata import remote_test_data_generator
from birdgame.visualization.animated_viz_predictions import animated_predictions_graph


class Quarantine:
    def __init__(self, horizon):
        self.horizon = horizon
        self.quarantine = []

    def add_to_quarantine(self, time, value):
        """ Adds a new value to the quarantine list. """
        self.quarantine.append((time + self.horizon, value))

    def pop_from_quarantine(self, current_time):
        """ Returns the most recent valid data point from quarantine. """
        valid = [(j, (ti, xi)) for (j, (ti, xi)) in enumerate(self.quarantine) if ti <= current_time]
        if valid:
            prev_ndx, (ti, prev_x) = valid[-1]
            self.quarantine = self.quarantine[prev_ndx:]  # Trim the quarantine list
            return prev_x
        return None


class TrackerBase(Quarantine):
    """
    Base class that handles quarantining of data points before they are eligible for processing.
    """

    def __init__(self, horizon: int):
        self.quarantine = []
        self.count = 0
        super().__init__(horizon)

    @abc.abstractmethod
    def tick(self, payload: dict):
        pass

    @abc.abstractmethod
    def predict(self) -> dict:
        pass

    def test_run(self, live=True, step_print=1000):
        from birdgame.model_benchmark.emwavartracker import EMWAVarTracker
        from birdgame.trackers.tracker_evaluator import TrackerEvaluator

        benchmark_tracker = EMWAVarTracker(horizon=self.horizon)
        my_run, bmark_run = TrackerEvaluator(self), TrackerEvaluator(benchmark_tracker)

        gen = live_data_generator() if live else remote_test_data_generator()
        try:
            for i, payload in enumerate(tqdm(gen)):

                my_run.tick_and_predict(payload)
                bmark_run.tick_and_predict(payload)

                if (i + 1) % step_print == 0:
                    print(f"My median score: {my_run.score():.4f} VS Benchmark median score: {bmark_run.score():.4f}")

            print(f"My median score: {my_run.score():.4f} VS Benchmark median score: {bmark_run.score():.4f}")
        except KeyboardInterrupt:
            print("Interrupted")

    def test_run_animated(self, live=True, window_size=50, from_notebook=False):
        from birdgame.model_benchmark.emwavartracker import EMWAVarTracker
        from birdgame.trackers.tracker_evaluator import TrackerEvaluator

        benchmark_tracker = EMWAVarTracker(horizon=self.horizon)
        my_run, bmark_run = TrackerEvaluator(self), TrackerEvaluator(benchmark_tracker)

        gen = live_data_generator() if live else remote_test_data_generator()

        use_plt_show = True if not from_notebook else False
        animated = animated_predictions_graph(gen, my_run, bmark_run, window_size=window_size, use_plt_show=use_plt_show)

        return animated
