import pandas as pd


def remote_test_data() -> pd.DataFrame:
    return pd.read_csv(
        'https://raw.githubusercontent.com/microprediction/birdgame/refs/heads/main/data/bird_feed_data.csv')


def remote_test_data_generator():
    """
      Generate the remote test data yielding one record (dict) at a time
    :return:
    """
    df = remote_test_data()
    for _, row in df.iterrows():
        yield row.to_dict()


if __name__ == '__main__':
    df = remote_test_data()
    print(df[:3])
