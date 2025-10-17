
from birdgame.examples.derived.gmmtracker import GMMTracker


if GMMTracker is not None:

    def test_gmm_test_run():
        tracker = GMMTracker()
        tracker.test_run(
            max_rows=1000,
            live=False, # Set to True to use live streaming data; set to False to use data from a CSV file
            step_print=1000 # Print the score and progress every 1000 steps
        )