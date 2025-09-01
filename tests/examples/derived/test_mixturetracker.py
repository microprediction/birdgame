from birdgame.examples.derived.mixturetracker import MixtureTracker
from birdgame.datasources.remotetestdata import remote_test_data_generator
from densitypdf import density_pdf

from birdgame.models.quantileregression import QuantileRegressionRiverTracker


def test_self():
    tracker = MixtureTracker()
    gen = remote_test_data_generator(start_time=0, max_rows=1000)
    for payload in gen:
        tracker.tick(payload, {})
        pdf_dict = tracker.predict()
        _ = density_pdf(pdf_dict, 1.0)
        if tracker.count > 100:
            break

    assert True

def test_run():
    tracker = QuantileRegressionRiverTracker()
    tracker.test_run(live=False, max_rows=1000)
