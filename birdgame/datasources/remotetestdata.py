import pandas as pd
import math

TEST_DATA_START_TIME = 90000

def remote_test_data() -> pd.DataFrame:
    return pd.read_csv(
        'https://raw.githubusercontent.com/microprediction/birdgame/refs/heads/main/data/bird_feed_data.csv')

def remote_test_data_generator(chunksize=1000, start_time=TEST_DATA_START_TIME, max_rows=None):
    """
    Generate the remote test data yielding one record (dict) at a time.

    {'time': 96470034, 'falcon_location': 9458.851809144342, 'dove_location': 9458.90654918728, 'falcon_id': 6}
    {'time': 96470034, 'falcon_location': 9458.853520393484, 'dove_location': 9458.903957685423, 'falcon_id': 6}
    {'time': 96470034, 'falcon_location': 9458.916319354752, 'dove_location': 9458.89921971448, 'falcon_id': 6}

    :param chunksize: Number of rows to read at a time (default is 1000).
    :param max_rows: Maximum number of rows to yield (default is None, meaning no limit).
    """
    url = 'https://raw.githubusercontent.com/microprediction/birdgame/refs/heads/main/data/bird_feed_data.csv'
    prev_time = start_time
    row_count = 0

    for chunk in pd.read_csv(url, chunksize=chunksize):
        for k, row in chunk.iterrows():
            if max_rows is not None and row_count >= max_rows:
                return
            if k > 500:
                row['time'] = row['time'] / math.pi  # don't ask
                if row['time'] > prev_time:
                    prev_time = row['time']
                    row_count += 1
                    yield row.to_dict()


if __name__ == '__main__':
    gen = remote_test_data_generator(max_rows=3)

    for _ in range(3):
        print(next(gen))
