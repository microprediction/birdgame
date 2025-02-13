from birdgame.datasources.remotetestdata import remote_test_data, remote_test_data_generator
import pandas as pd
import pytest 


def test_remote_test_data():
    df = remote_test_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_remote_test_data_generator():
    generator = remote_test_data_generator()
    first_record = next(generator)
    assert isinstance(first_record, dict)
    assert len(first_record) > 0
    assert 'time' in first_record


if __name__ == '__main__':
    pytest.main([__file__])
