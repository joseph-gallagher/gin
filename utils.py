import numpy as np
from random import shuffle, choice
import itertools
from math import inf
import os
from time import time


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def powerset(iterable):

    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))


def value(card):
    if card[0] < 10:
        return card[0]
    else:
        return 10


def is_deadwood(card, hand):
    """
    Checks to see if a card is deadwood in a given hand.

    Examples:

    #>>> hand = [(1, 'H'), (2,'H'), (3, 'H'), (7, 'C')]
    #>>> card = [(1, 'H')]
    #>>> is_deadwood(card, hand)
    False
    #>>> card = [(7, 'C')]
    #>>> is_deadwood(card, hand)
    True

    :param card: card
    :type card: list
    :param hand: hand
    :type hand: list
    :return: True if card exists in no melds, False otherwise
    :rtype: bool
    :raises ValueError: If the given card is not in the given hand.
    """

    if card not in hand:
        raise ValueError('The card argument should be an element of the hand argument')

    rank = card[0]
    suit = card[1]

    same_rank = [card for card in hand if card[0] == rank]
    if len(same_rank) >= 3:
        return False

    same_suit = [card for card in hand if card[1] == suit]
    sorted(same_suit, key=lambda x: x[0])

    i = same_suit.index(card)
    suit_run = []
    for j in range(len(same_suit)):
        card_2 = same_suit[j]
        if card_2[0] == (rank + (j - i)):
            suit_run.append(card_2)

    if len(suit_run) >= 3:
        return False
    else:
        return True


def melds_in(card, hand):
    """
    Gives all melds within hand that contain card.

    Examples:

    #>>> hand = [(2, 'H'), (3, 'H'), (4, 'H'), (3, 'S'), (3, 'C'), (8, 'D')]
    #>>> card = (3, 'H')
    #>>> melds_in(card, hand)
    [[(3, 'H'), (3, 'S'), (3, 'C')], [(2, 'H'), (3, 'H'), (4, 'H')]]
    #>>> card = (2, 'H')
    #>>> melds_in(card, hand)
    [[(2, 'H'), (3, 'H'), (4, 'H')]]
    #>>> card = (8, 'D')
    #>>> melds_in(card, hand)
    []

    :param card: card
    :type card: tuple
    :param hand: hand
    :type hand: list
    :return: All melds within hand containing card
    :rtype: list
    :raises ValueError: If the given card is not part of the given hand.

    """

    if card not in hand:
        raise ValueError('The card argument should be an element of the hand argument')

    rank = card[0]
    suit = card[1]

    melds = []

    same_rank = [card for card in hand if card[0] == rank]
    same_suit = [card for card in hand if card[1] == suit]
    sorted(same_suit, key=lambda x: x[0])

    # If there are enough of the same rank, return same rank melds:
    for j in range(3, len(same_rank) + 1):
        same_j = itertools.combinations(same_rank, j)
        same_j = [list(g) for g in same_j if card in g]
        melds.extend(same_j)

    # Find the maximal run through the given card
    i = same_suit.index(card)
    suit_run = []
    for j in range(len(same_suit)):
        card_2 = same_suit[j]
        if card_2[0] == (rank + (j - i)):
            suit_run.append(card_2)

    # Add all runs through the given card
    i = suit_run.index(card)
    for j in range(len(suit_run)):
        for k in range(j):
            if k <= i <= j and j - k > 1:
                melds.append(suit_run[k:j + 1])

    return melds


def min_loose_points(hand):
    """
    Computes the smallest possible deadwood value in a hand, and returns an optimal meld.

    Examples:

    """

    deadwood = []
    meldable = []

    best_meld = []

    for card in hand:
        if is_deadwood(card, hand):
            deadwood.append(card)
        else:
            meldable.append(card)

    deadwood_pts = sum([value(card) for card in deadwood])

    # If all cards are deadwood, can do no melding:
    if meldable == []:
        return [deadwood_pts, best_meld]


    # The first card is in at most one of the melds it can participate in.
    first_card = meldable[0]
    first_melds = melds_in(first_card, meldable)
    current_best_deadwood = inf
    best_meld_with_first = []
    for meld in first_melds:
        remaining = [card for card in meldable if card not in meld]
        dw_with_this_meld = min_loose_points(remaining)[0] + deadwood_pts
        best_meld_of_remaining = min_loose_points(remaining)[1]
        if dw_with_this_meld < current_best_deadwood:
            current_best_deadwood = dw_with_this_meld
            best_meld_of_remaining.append(meld)
            best_meld_with_first = best_meld_of_remaining

    # We must take into account that the best meld may not involve the first
    # card at all:
    remaining = meldable[1:]
    dw_without_first = min_loose_points(remaining)[0] + deadwood_pts + value(first_card)
    best_meld_without_first = min_loose_points(remaining)[1]

    if dw_without_first < current_best_deadwood:
        current_best_deadwood = dw_without_first
        best_meld = best_meld_without_first
    else:
        best_meld = best_meld_with_first

    return [current_best_deadwood, best_meld]

def layoffable(melded_hand):

    """
    Computes a list of all cards which can be laid off on the given meld. May contain duplicates.

    :param melded_hand: A meld as generated by melds_in.
    :type melded_hand: List
    :return: A list of cards which can be added to the given melds.
    """

    possible = []
    for meld in melded_hand:
        ranks = [card[0] for card in meld]
        ranks = list(set(ranks))
        if len(ranks) == 1:
            rank = ranks[0]
            same_rank = [(rank, 'H'), (rank, 'C'), (rank, 'D'), (rank, 'S')]
            for card in meld:
                same_rank.remove(card)
            possible.extend(same_rank)
        else:
            suit = meld[0][1]
            max_rank = max(ranks)
            min_rank = min(ranks)
            if max_rank < 13:
                possible.append((max_rank+1, suit))
            if min_rank > 1:
                possible.append((min_rank-1, suit))

    return possible

'''
A class representing a deck
'''


class Deck:

    def __init__(self):

        ranks = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        suits = ['H', 'D', 'C', 'S']
        '''
        The values of the dictionary have the following meanings:
            0 - in the deck
            -1 - in Player 1's hand
            1 - in Player 2's hand
            -0.5 - In discard pile, discarded by Player 1
            0.5 - In discard pile, discarded by Player 2
        '''
        self.game_state = {(r, s): 0 for s in suits for r in ranks}

        self.deck_order = [(r, s) for s in suits for r in ranks]
        shuffle(self.deck_order)

        # A kluge: start the game by pretending that Player 2
        # has discarded the top card in the discard.
        self.top_discard = self.deck_order.pop()
        self.game_state[self.top_discard] = 0.5

        self.turn = 1

        self.knock = False

    def deal(self, player_1, player_2):

        for i in range(10):
            c_1 = self.deck_order.pop()
            player_1.hand.append(c_1)
            self.game_state[c_1] = -1

            c_2 = self.deck_order.pop()
            player_2.hand.append(c_2)
            self.game_state[c_2] = 1

    def draw_deck(self):

        drawn = self.deck_order.pop()
        if self.turn % 2 == 1:
            self.game_state[drawn] = -1
        else:
            self.game_state[drawn] = 1

        return drawn

    def draw_discard(self):

        if self.turn % 2 == 1:
            self.game_state[self.top_discard] = -1
        else:
            self.game_state[self.top_discard] = 1

        return self.top_discard

    def take_discard(self, card):

        if self.turn % 2 == 1:
            self.game_state[card] = -0.5
        else:
            self.game_state[card] = 0.5

        self.top_discard = card



# A test of the current capabilities, to make sure we haven't broken anything:
'''
deck = Deck()
player_1 = NNPlayer()
player_2 = Greedy_Player()
game = TrainingGame(deck, player_1, player_2)
game.play_game(trace=True)

current_dir = os.getcwd()
first_try = 'gyeah'
save_path = os.path.join(current_dir, first_try)
save_path = os.path.join(save_path, first_try)
print(player_1.graph.path)
player_1.graph.save_state(save_path=save_path)
print(player_1.graph.path)



def time_check(n):

    t_1 = time()
    player_1 = NNPlayer()
    player_2 = NNPlayer()
    for i in range(n):
        deck = Deck()
        game = Game(deck, player_1, player_2)
        game.play_game()
    t_2 = time()
    print('Took ' + str(t_2-t_1))

time_check(5)
time_check(10)
# time_check(50)

'''


