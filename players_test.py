from players import *
import unittest


class DrawDeckTest(unittest.TestCase):

    def setUp(self):
        self.deck = Deck()
        self.player_1 = Player()
        self.player_2 = Player()

    def tearDown(self):
        self.deck = None
        self.player_1 = None

    def test_updates_states(self):
        top_card = self.deck.deck_order[-1]
        self.player_1.draw_deck(self.deck)
        self.assertEqual(-1, self.deck.game_state[top_card])

    def test_updates_states_2(self):
        self.deck.turn = 2
        top_card = self.deck.deck_order[-1]
        self.player_2.draw_deck(self.deck)
        self.assertEqual(1, self.deck.game_state[top_card])

    def test_ends_in_hand(self):
        top_card = self.deck.deck_order[-1]
        self.player_1.draw_deck(self.deck)
        self.assertIn(top_card, self.player_1.hand)


class DrawDiscardTest(unittest.TestCase):

    def setUp(self):
        self.deck = Deck()
        self.player_1 = Player()
        self.player_2 = Player()

    def test_updates_states_1(self):
        top_discard = self.deck.top_discard
        self.player_1.draw_discard(self.deck)
        self.assertEqual(-1, self.deck.game_state[top_discard])

    def test_updates_states_2(self):
        self.deck.turn = 2
        top_discard = self.deck.top_discard
        self.player_2.draw_discard(self.deck)
        self.assertEqual(1, self.deck.game_state[top_discard])

    def test_ends_in_hand(self):
        top_discard = self.deck.top_discard
        self.player_1.draw_discard(self.deck)
        self.assertIn(top_discard, self.player_1.hand)

