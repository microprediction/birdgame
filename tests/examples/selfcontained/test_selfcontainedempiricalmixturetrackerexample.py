from birdgame.examples.selfcontained.selfcontainedmixturetrackerexample import SelfContainedMixtureTrackerExample
from birdgame.datasources.remotetestdata import remote_test_data_generator
from densitypdf import density_pdf


def test_selfcontained():
    tracker = SelfContainedMixtureTrackerExample()
    gen = remote_test_data_generator(start_time=0)
    for payload in gen:
        tracker.tick(payload)
        pdf_dict = tracker.predict()
        _ = density_pdf(pdf_dict, 1.0)
        if tracker.count > 100:
            break

    assert True
