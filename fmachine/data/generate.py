import pprint
from typing import List, Dict, Iterable, Tuple, Callable, Any, Union

import numpy as np
import pandas as pd
from scipy.sparse.coo import coo_matrix


class IndexMap(dict):
    """Class to handle indexing with a set range, with special unknown return."""
    def __init__(self, *args, unseen_index: int=0) -> None:
        self.unseen_index = unseen_index
        super().__init__(*args)

    def __missing__(self, key) -> Any:
        if key not in self.keys():
            return self.unseen_index

    def __len__(self) -> int:
        return len(self.keys()) + 1

    def __repr__(self) -> str:
        return f"IndexMap({super().__repr__()}, unseen_index={self.unseen_index})"

    @classmethod
    def create_from_iterable(cls, items: Iterable,
                             unseen_index: int=0) -> "IndexMap":

        # Get uniques
        unique_items = set(list(items))
        if len(items) == len(unique_items):
            # If the list was already unique, maintain the original ordering
            unique_items = items

        available_indexes = list(range(len(unique_items) + 1))
        _ = available_indexes.pop(available_indexes.index(unseen_index))

        return IndexMap({k: v for k, v in zip(unique_items, available_indexes)},
                        unseen_index=unseen_index)

    def inverse(self) -> Dict[Any, int]:
        return {v: k for k, v in self.items()}


def rate_or_not(rating_f: Callable,
                p: float=0.3,
                null_rating: Any=0, **kwargs) -> Union[int, Any]:
    """
    Return rating from rating_f with probability p, else no rating.

    :param rating_f: Function to use for rating. Default 1-5 star rating.
    :param p: Probability of returning a rating.
    :param null_rating: Value to return when a rating isn't made.
    """

    # If null_rating is str, will cast other outputs if used directly here
    r = np.random.choice((rating_f(**kwargs), np.nan),
                         p=(1 - p, p))

    return null_rating if np.isnan(r) else r


def star_rating(bias: float=1.0) -> int:
    return int(min(5, max(1, np.random.normal(2.5, 2.5) * bias)))


class Item:
    """Class handling item ID (and later meta data)."""
    def __init__(self,
                 id: str='item') -> None:
        self.id = id

    def __repr__(self) -> str:
        return f"Item(id={self.id})"

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and (self.id == other.id)


class Items(IndexMap):
    """Map of items. Nothing additional yet."""
    pass


class UserProfile:
    """User profile has preferences for items. (And later meta data"""
    def __init__(self, perfs: Dict[Item, int]):
        """
        :param perfs: User's preferences (biases) for items, in dict indexed by items.
        """
        self.perfs = perfs

    def __repr__(self) -> str:
        return f"UserProfile(perfs={self.perfs})"


class User:
    """User has unique id, ratings, and preferences."""
    def __init__(self,
                 profile: UserProfile,
                 density: float=0.3,
                 rating_f: Callable=star_rating,
                 missing_rating=np.nan,
                 id: Any='user') -> None:
        """
        :param profile: UserProfile describing preferences for known items.
        :param density: Density of user ratings (probability user rates a given item)
        :param rating_f: Function to use to generate ratings. Default star_rating.
        :param missing_rating: Value to return for unrated items. Default 0.
        :param id: Name/id.
        """

        self.id = id
        self.profile = profile
        self.density = density
        self.rating_f = rating_f
        self.missing_rating = missing_rating

        self.ratings: Dict[Item, int] = None
        self._rate()

    def _rate(self) -> None:
        self.ratings = {i: rate_or_not(rating_f=self.rating_f,
                                       null_rating=0,
                                       p=self.density) for i, p in self.profile.perfs.items()}

    def __repr__(self) -> str:
        return f"User(id={self.id}, profile={self.profile}, density={self.density})"

    def __hash__(self) -> int:
        return hash((self.__repr__()))

    def __eq__(self, other) -> bool:
        return (isinstance(other, type(self))
                and (self.id, self.profile, self.density) == (other.id, other.profile, self.density))


class SparseUIM(coo_matrix):
    """Class to handle user-item matrixes in sparse format, and convert to other formats."""
    @classmethod
    def create_from_coords(cls, coords: List[Tuple[int, Tuple[int, int]]]) -> "SparseUIM":

        r = [c[0] for c in coords]
        u = [c[1][0] for c in coords]
        i = [c[1][1] for c in coords]

        return SparseUIM((r, (u, i)))

    def __getitem__(self, index) -> Any:
        return self.tocsr()[index[0], index[1]]

    def todf(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame.sparse.from_spmatrix(self, *args, **kwargs)


class Users(IndexMap):
    """Map of users. Nothing additional yet."""
    pass


class UsersItems:
    """Class to handle user-item matrix"""
    def __init__(self, ratings_df) -> None:

        self.user_map = IndexMap.create_from_iterable(ratings_df.user_id)
        self.item_map = IndexMap.create_from_iterable(ratings_df.item_id)
        ratings_df.loc[:, 'user_id'] = ratings_df.user_id.map(self.user_map)
        ratings_df.loc[:, 'item_id'] = ratings_df.user_id.map(self.item_map)

        self._uim = SparseUIM.create_from_coords(self.tocoords())

    def __repr__(self) -> str:
        return f"UsersItems({self.users}, {self.items})"

    def __str__(self) -> str:
        return pprint.pformat(self.__repr__())

    @property
    def uim(self) -> SparseUIM:
        """The user-item matrix and SparseUIM"""
        return self._uim

    def tocoords(self) -> List[Tuple[Any, Tuple[int, int]]]:
        """Convert the users and items to coordinate tuples [(r1, (i1, j1)), (r2, (i2, j2)), ... ]"""
        coords = [(r, (uk, self.items[ik]))
                  for uk, uv in self.users.inverse().items() for ik, r in uv.ratings.items()]
        return coords

    def __getitem__(self, index) -> Any:
        return self.uim[index[0], index[1]]

    def get_ui(self, user: User, item: Item) -> Any:
        """Index using User and Item objects."""
        i = self.users[user]
        j = self.items[item]

        return self.uim[i, j]

    def todf(self) -> pd.DataFrame:
        """
        Convert to DataFrame containing SparseArrays.
        TODO: Weird behaviour in some cases - see unittest.
        """
        df = self.uim.todf(index=pd.Series(['unknown'] + [u.id for u in self.users.keys()],
                                           name='users'),
                           columns=pd.Series(['unknown'] + [i.id for i in self.items.keys()],
                                             name='items'))

        return df
