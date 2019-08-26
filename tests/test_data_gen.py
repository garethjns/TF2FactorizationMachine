import unittest
from typing import List, Tuple

import numpy as np
from scipy.sparse.coo import coo_matrix

from fmachine.data_gen import IndexMap, SparseUIM, Item, UserProfile, User, UsersItems, star_rating, rate_or_not


def make_some_items(ids=None) -> List[Item]:
    if ids is None:
        ids = [1, 2, 3]

    return [Item(id=i) for i in ids]


def make_some_users() -> Tuple[List[Item], List[UserProfile], List[User]]:
    i1, i2, i3 = make_some_items()

    profile_1 = UserProfile(perfs={i1: 5, i2: 3, i3: 1})
    profile_2 = UserProfile(perfs={i1: 1, i2: 3, i3: 5})
    profile_3 = UserProfile(perfs={i1: 1, i2: 1, i3: 1})
    profile_4 = UserProfile(perfs={i1: 5, i2: 5, i3: 5})
    profiles = [profile_1, profile_2, profile_3, profile_4]

    users = [User(id=e,
                  profile=p) for e, p in enumerate(profiles)]

    return [i1, i2, i3], profiles, users


class TestIndexMap(unittest.TestCase):
    def test_create_from_list(self):
        im = IndexMap.create_from_iterable(['a', 'b', 'c'])
        self.assertDictEqual(im, {'a': 1,
                                  'b': 2,
                                  'c': 3})

    def test_create_from_array(self):
        im = IndexMap.create_from_iterable(np.array(['a', 'b', 'c']))
        self.assertDictEqual(im, {'a': 1,
                                  'b': 2,
                                  'c': 3})

    def test_inverse(self):
        im = IndexMap.create_from_iterable(['a', 'b', 'c'])
        self.assertDictEqual(im.inverse(), {1: 'a',
                                            2: 'b',
                                            3: 'c'})

    def test_unseen_index(self):
        im = IndexMap.create_from_iterable(['a', 'b', 'c'])
        self.assertEqual(im['d'], 0)

    def test_alternative_unseen_index(self):
        im = IndexMap.create_from_iterable(['a', 'b', 'c'],
                                           unseen_index=1)
        self.assertEqual(im['d'], 1)


class TestSparseUIM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.coords_1 = [(1, (1, 1)), (2, (2, 2)), (3, (3, 3)), (4, (4, 4)), (5, (5, 5))]
        cls.coo_1 = coo_matrix(([1, 2, 3, 4, 5], ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])))
        cls.coords_2 = [(1, (2, 3)), (2, (1, 2)), (3, (3, 3)), (4, (3, 2)), (5, (1, 3)), (6, (2, 4))]
        cls.coo_2 = coo_matrix(([1, 2, 3, 4, 5, 6], ([2, 1, 3, 3, 1, 2], [3, 2, 3, 2, 3, 4])))

    def test_create_from_coords(self):

        uim = SparseUIM.create_from_coords(self.coords_1)

        self.assertTrue(np.all(self.coo_1.toarray() == uim.toarray()))
        self.assertTrue((self.coo_1 != uim).sum() == 0)

    def test_create_from_coords_2(self):
        uim = SparseUIM.create_from_coords(self.coords_2)

        self.assertTrue(np.all(self.coo_2.toarray() == uim.toarray()))
        self.assertTrue((self.coo_2 != uim).sum() == 0)

    def test_unseen_entries(self):
        uim = SparseUIM.create_from_coords(self.coords_1)

        self.assertTrue(uim[0, 0] == 0)
        self.assertTrue(uim[1, 0] == 0)
        self.assertTrue(uim[0, 1] == 0)

    def test_indexing_1(self):
        uim = SparseUIM.create_from_coords(self.coords_1)

        for (v, (i, j)) in self.coords_1:
            self.assertTrue(uim[i, j], v)

    def test_indexing_2(self):
        uim = SparseUIM.create_from_coords(self.coords_2)

        for (v, (i, j)) in self.coords_2:
            self.assertTrue(uim[i, j], v)

    def test_todf(self):
        uim = SparseUIM.create_from_coords(self.coords_1)
        self.assertTrue(np.all(uim.todf().values == uim.toarray()))

        uim = SparseUIM.create_from_coords(self.coords_2)
        self.assertTrue(np.all(uim.todf().values == uim.toarray()))


class TestItem(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.i1, cls.i2, cls.i3, cls.i4 = make_some_items(ids=[1, 'item_2', 3, 1])

    def test_eq(self):
        self.assertEqual(self.i1, self.i1)
        self.assertEqual(self.i1, self.i4)

        self.assertNotEqual(self.i1, self.i2)
        self.assertNotEqual(self.i1, self.i3)
        self.assertNotEqual(self.i2, self.i3)

    def test_index(self):
        users_dict = {self.i1: 1, self.i2: 2, self.i3: 3}
        self.assertEqual(users_dict[self.i1], 1)
        self.assertEqual(users_dict[self.i2], 2)
        self.assertEqual(users_dict[self.i3], 3)
        self.assertEqual(users_dict[self.i4], 1)

    def test_in_index_map(self):
        users_map = IndexMap.create_from_iterable([self.i1, self.i2, self.i3])
        self.assertEqual(users_map[self.i1], 1)
        self.assertEqual(users_map[self.i2], 2)
        self.assertEqual(users_map[self.i3], 3)
        self.assertEqual(users_map[self.i4], 1)


class TestUserProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, cls.profiles, _ = make_some_users()


class TestUser(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _, _, cls.users = make_some_users()

    def test_rate(self):
        self.users[0]._rate()
        self.users[1]._rate()

    def test_star_rating(self):
        for _ in range(100):
            r = star_rating()
            self.assertTrue((r >= 1) & (r <= 5))

    def test_rate_or_not(self):
        for _ in range(100):
            r = rate_or_not(rating_f=star_rating)
            self.assertTrue((r >= 0) & (r <= 5))

            r = rate_or_not(rating_f=star_rating,
                            null_rating='missing')
            self.assertTrue(r in ['missing'] + list(range(1, 6)))


class TestUsersItems(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        items, _, users = make_some_users()

        cls.users = IndexMap.create_from_iterable(users)
        cls.items = IndexMap.create_from_iterable(items)

        cls.ui = UsersItems(users=cls.users,
                            items=cls.items)

    @unittest.skip('Todo')
    def test_uim(self):
        self.ui.uim

    @unittest.skip('Bug?!')
    def test_todf(self):
        """This fails...."""
        self.assertTrue(np.all(self.ui.todf().values == self.ui.uim.toarray()))

    @unittest.skip('Todo')
    def test_eval(self):
        pass

    @unittest.skip('Todo')
    def test_indexing(self):
        pass

    @unittest.skip('Todo')
    def test_indexing_with_objects(self):
        pass

