from birdgame.datasources.livedata import live_data_generator
import pytest


def test_live_data_generator():
    generator = live_data_generator()
    first_record = next(generator)
    assert isinstance(first_record, dict)
    assert len(first_record) > 0
    assert 'time' in first_record


if __name__ == '__main__':
    pytest.main([__file__])
