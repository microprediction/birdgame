import pandas as pd

from birdgame.animation.animatebirds import START_TIME

START_TIME = 90000

def remote_test_data() -> pd.DataFrame:
    return pd.read_csv(
        'https://raw.githubusercontent.com/microprediction/birdgame/refs/heads/main/data/bird_feed_data.csv')


def remote_test_data_generator(chunksize=1000):
    """
    Generate the remote test data yielding one record (dict) at a time.

    :param chunksize: Number of rows to read at a time (default is 1000).
    """
    url = 'https://raw.githubusercontent.com/microprediction/birdgame/refs/heads/main/data/bird_feed_data.csv'
    for chunk in pd.read_csv(url, chunksize=chunksize):
        for k, row in chunk.iterrows():
            if row['time']>START_TIME:
                yield row.to_dict()


if __name__ == '__main__':
    gen = remote_test_data_generator()

    # Print the first 3 records
    for _ in range(3):
        print(next(gen))

if __name__ == '__main__':
    gen = remote_test_data_generator()

    for _ in range(3):
        print(next(gen))