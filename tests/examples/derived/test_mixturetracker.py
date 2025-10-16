from birdgame.examples.derived.mixturetracker import MixtureTracker
from birdgame.datasources.remotetestdata import remote_test_data_generator
from densitypdf import density_pdf


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
