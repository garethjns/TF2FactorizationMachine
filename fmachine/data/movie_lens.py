import pandas as pd


def load_move_lens_100k(path: str):

    load_args = {'sep':'\t',
                 'header': None,
                 'names': ['user_id', 'item_id', 'rating', 'timestamp']}

    try:
        data = pd.read_csv(path, **load_args)
    except FileNotFoundError:
        data = pd.read_csv('../' + path, **load_args)

    return data


if __name__ == "__main__":

    train = load_move_lens_100k(path='../../data/ml-100k/ua.base')
    test = load_move_lens_100k(path='../../data/ml-100k/ua.test')