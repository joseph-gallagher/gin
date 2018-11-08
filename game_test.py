import unittest
from game import *


class ScoreTest(unittest.TestCase):

    def setUp(self):
        self.player_1 = Player()
        self.player_2 = Player()
        self.deck = Deck()

    def tearDown(self):
        self.player_1 = None
        self.player_2 = None
        self.deck = None

    def test_has_gin(self):

        game = Game(self.deck, self.player_1, self.player_2)
        game.player_1.hand = [(1, 'H'), (2, 'H'), (3, 'H'), (7, 'H'), (7, 'C'), (7, 'S'), (7, 'D'),
                         (11, 'C'), (12, 'C'), (13, 'C')]
        game.player_2.hand = [(1, 'C'), (2, 'C'), (3, 'C'), (8, 'H'), (8, 'C'), (8, 'D'), (5, 'S'),
                         (11, 'D'), (12, 'D'), (13, 'D')]
        # Technically, this won't even be read, though we include it for completion.
        game.deck.ending_player_melds = [[(1, 'H'), (2, 'H'), (3, 'H')],
                                         [(7, 'H'), (7, 'C'), (7, 'S'), (7, 'D')],
                                         [(11, 'C'), (12, 'C'), (13, 'C')]]
        self.assertEqual(30, game.score_game(game.player_1, game.player_2))

    def test_can_layoff_and_cut(self):

        game = Game(self.deck, self.player_1, self.player_2)
        game.player_1.hand = [(1, 'H'), (2, 'H'), (3, 'H'), (5, 'H'), (5, 'D'), (5, 'C'),
                              (7, 'C'), (7, 'D'), (7, 'S'), (9, 'D')]
        game.player_2.hand = [(6, 'H'), (6, 'D'), (6, 'S'), (8, 'H'), (8, 'D'), (8, 'S'),
                              (9, 'H'), (9, 'D'), (9, 'S'), (4, 'H')]
        game.deck.ending_player_melds = [[(1, 'H'), (2, 'H'), (3, 'H')],
                                         [(5, 'H'), (5, 'D'), (5, 'C')],
                                         [(7, 'C'), (7, 'D'), (7, 'S')]]
        self.assertEqual(-34, game.score_game(game.player_1, game.player_2))

    def test_can_layoff_no_cut(self):

        game = Game(self.deck, self.player_1, self.player_2)
        game.player_1.hand = [(1, 'H'), (2, 'H'), (3, 'H'), (5, 'H'), (5, 'D'), (5, 'C'),
                              (7, 'C'), (7, 'D'), (7, 'S'), (9, 'D')]
        game.player_2.hand = [(6, 'H'), (6, 'D'), (6, 'S'), (8, 'H'), (8, 'D'), (8, 'S'),
                              (9, 'H'), (9, 'D'), (13, 'S'), (4, 'H')]
        game.deck.ending_player_melds = [[(1, 'H'), (2, 'H'), (3, 'H')],
                                         [(5, 'H'), (5, 'D'), (5, 'C')],
                                         [(7, 'C'), (7, 'D'), (7, 'S')]]
        self.assertEqual(19, game.score_game(game.player_1, game.player_2))

    def test_no_layoff_and_cut(self):

        game = Game(self.deck, self.player_1, self.player_2)
        game.player_1.hand = [(1, 'H'), (2, 'H'), (3, 'H'), (5, 'H'), (5, 'D'), (5, 'C'),
                              (7, 'C'), (7, 'D'), (7, 'S'), (9, 'D')]
        game.player_2.hand = [(6, 'H'), (6, 'D'), (6, 'S'), (8, 'H'), (8, 'D'), (8, 'S'),
                              (9, 'H'), (9, 'D'), (9, 'S'), (2, 'C')]
        game.deck.ending_player_melds = [[(1, 'H'), (2, 'H'), (3, 'H')],
                                         [(5, 'H'), (5, 'D'), (5, 'C')],
                                         [(7, 'C'), (7, 'D'), (7, 'S')]]
        self.assertEqual(-32, game.score_game(game.player_1, game.player_2))


# class PlayTest(unittest.TestCase):


# class TrainTest(unittest.TestCase):

    None

if __name__ == '__main__':
    unittest.main()


