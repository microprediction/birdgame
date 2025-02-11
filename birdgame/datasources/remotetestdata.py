import pandas as pd


def remote_test_data()->pd.DataFrame:
    return pd.read_csv('')


if __name__=='__main__':
    df = remote_test_data()
    print(df[:3])