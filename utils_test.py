import unittest
from utils import *


class DeadwoodTest(unittest.TestCase):

    def test_no_meld(self):
        hand = [(1, 'H'), (2, 'H'), (3, 'H'), (5, 'S')]
        card = (5, 'S')
        self.assertTrue(is_deadwood(card, hand))

    def test_one_meld(self):
        hand = [(1, 'H'), (2, 'H'), (3, 'H'), (5, 'S')]
        card = (2, 'H')
        self.assertFalse(is_deadwood(card, hand))

    def test_two_melds(self):
        hand = [(1, 'H'), (2, 'H'), (3, 'H'), (3, 'C'), (3, 'S')]
        card = (3, 'H')
        self.assertFalse(is_deadwood(card, hand))

    def test_not_in_hand(self):
        hand = [(1, 'H'), (2, 'H'), (3, 'H'), (5, 'S')]
        card = (2, 'C')
        with self.assertRaises(ValueError):
            is_deadwood(card, hand)


class MeldsTest(unittest.TestCase):

    def test_no_meld(self):
        hand = [(1,'H'), (2, 'H'), (3, 'H'), (5, 'S')]
        card = (5, 'S')
        self.assertEqual([], melds_in(card, hand))

    def test_one_meld(self):
        hand = [(1,'H'), (2, 'H'), (3, 'H'), (5, 'S')]
        card = (2, 'H')
        self.assertEqual([[(1,'H'), (2, 'H'), (3, 'H')]], melds_in(card, hand))

    def test_two_melds(self):
        hand = [(1,'H'), (2, 'H'), (3, 'H'), (3, 'C'), (3, 'S')]
        card = (3, 'H')
        self.assertEqual([[(3, 'H'), (3, 'C'), (3, 'S')], [(1, 'H'), (2, 'H'), (3, 'H')]],
                         melds_in(card, hand))

    def test_not_in_hand(self):
        hand = [(1, 'H'), (2, 'H'), (3, 'H'), (5, 'S')]
        card = (2, 'C')
        with self.assertRaises(ValueError):
            melds_in(card, hand)


class MinLooseTest(unittest.TestCase):

    def test_have_gin(self):
        hand = [(1, 'H'), (2, 'H'), (3, 'H'), (8, 'H'), (8, 'C'), (8, 'S'),
                (10, 'D'), (11, 'D'), (12, 'D'), (13, 'D')]
        self.assertEqual(0, min_loose_points(hand)[0])

    def test_last_in_twos(self):
        hand = [(1, 'H'), (2, 'H'), (3, 'H'), (8, 'H'), (8, 'C'), (8, 'S'),
                (10, 'D'), (10, 'C'), (12, 'D'), (12, 'C')]
        self.assertEqual(40, min_loose_points(hand)[0])

    def test_melds_intersect(self):
        hand = [(3, 'H'), (4, 'H'), (5, 'H'), (5, 'C'), (5, 'S')]
        self.assertEqual(7, min_loose_points(hand)[0])


class LayoffTest(unittest.TestCase):

    def test_no_layoff(self):
        melded_hand = [[(7, 'H'), (7, 'C'), (7, 'D'), (7, 'S')],
                       [(8, 'H'), (8, 'C'), (8, 'D'), (8, 'S')]]
        self.assertEqual([], layoffable(melded_hand))

    def test_runs(self):
        melded_hand = [[(1, 'H'), (2, 'H'), (3, 'H')], [(7, 'C'), (8, 'C'), (9, 'C')],
                       [(11, 'D'), (12, 'D'), (13, 'D')]]
        self.assertEqual(set([(4, 'H'), (6, 'C'), (10, 'C'), (10, 'D')]),
                         set(layoffable(melded_hand)))

    def test_ranks(self):
        melded_hand = [[(1, 'H'), (1, 'D'), (1, 'S')], [(3, 'H'), (3, 'C'), (3, 'S')],
                       [(5, 'H'), (5, 'C'), (5, 'S'), (5, 'D')]]
        self.assertEqual(set([(1, 'C'), (3, 'D')]), set(layoffable(melded_hand)))

    def test_intersect(self):
        melded_hand = [[(1, 'H'), (2, 'H'), (3, 'H')], [(4, 'C'), (4, 'D'), (4, 'S')]]
        self.assertEqual(set([(4, 'H')]), set(layoffable(melded_hand)))


if __name__ == '__main__':
    unittest.main()
