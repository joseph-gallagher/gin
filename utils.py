import numpy as np
from random import shuffle, choice
import itertools
from math import inf
from simple_nn import SimpleGraph
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


class Player:

    def __init__(self):

        self.hand = []

    def draw_deck(self, deck):

        drawn = deck.draw_deck()
        self.hand.append(drawn)

    def draw_discard(self, deck):

        drawn = deck.draw_discard()
        self.hand.append(drawn)

    def discard(self, card, deck):

        self.hand.remove(card)
        deck.take_discard(card)

    def knock(self, card, deck):

        self.hand.remove(card)
        deck.take_discard(card)
        deck.knock = True
        return card

    # A completely random play method:
    def play(self, deck):

        from_deck = choice([True, False])
        if from_deck:
            self.draw_deck(deck)
        else:
            self.draw_discard(deck)

        to_discard = choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        self.discard(self.hand[to_discard], deck)


class Greedy_Player(Player):

    def __init__(self):

        super(Greedy_Player, self).__init__()

    # A greedy algorithm that takes the top discard if it lowers
    # the deadwood value, otherwise draws from the deck.
    def play(self, deck):

        cur_deadwood = min_loose_points(self.hand)[0]
        # Once we get below 3 loose points, knock:
        if cur_deadwood < 3:
            optimal_meld = min_loose_points(self.hand)[1]
            deck.knock = True
            deck.ending_player_melds = optimal_meld
            return

        top_discard = deck.top_discard
        min_with_top_discard = cur_deadwood
        to_discard_next = 10

        for i in range(10):
            possible_hand = self.hand[:i]
            possible_hand.extend(self.hand[i + 1:])
            possible_hand.append(top_discard)
            m = min_loose_points(possible_hand)[0]
            if m < min_with_top_discard:
                min_with_top_discard = m
                to_discard_next = i

        if min_with_top_discard < cur_deadwood:
            self.draw_discard(deck)
            self.discard(self.hand[to_discard_next], deck)
            return
        else:
            self.draw_deck(deck)
            to_discard_next = 10
            lowest_deadwood = cur_deadwood
            for i in range(11):
                possible_hand = self.hand[:i]
                possible_hand.extend(self.hand[i + 1:])
                m = min_loose_points(possible_hand)[0]
                if m < lowest_deadwood:
                    lowest_deadwood = m
                    to_discard_next = i

            self.discard(self.hand[to_discard_next], deck)


class NNPlayer(Player):

    def __init__(self, is_player_1=True, path=None, training=False):

        super(NNPlayer, self).__init__()
        self.graph = SimpleGraph(path)
        self.is_player_1 = is_player_1
        self.training = training

    def flatten(self, state):

        # Represents the current deck.state as a one hot vector for feeding into graph.
        # Note: This crucially depends on the property that dicts are implicitly ordered in
        # Python! This only holds for Python 3.6 and higher.
        one_hot = [state[k] for k in state]
        return one_hot

    def discard_strategy(self, deck):

        state = deck.game_state.copy()
        cur_hand = self.hand[:]
        top_discard = deck.top_discard
        (best_w_disc, to_discard_w_disc) = (inf, None)

        if self.is_player_1:

            state[top_discard] = -1
            cur_hand.append(top_discard)
            for card in cur_hand:

                # See what it would be like if you discarded it:
                state[card] = -0.5
                one_hot = self.flatten(state)
                evaluation = self.graph.evaluate(one_hot)
                if evaluation == inf:
                    raise ValueError('The neural-net is returning infinite values (in discard_strategy)')
                if evaluation < best_w_disc:
                    (best_w_disc, to_discard_w_disc) = (evaluation, card)

                # Put it back for the next loop
                state[card] = -1

        else:

            state[top_discard] = 1
            cur_hand.append(top_discard)
            for card in cur_hand:

                # See what it would be like if you discarded it:
                state[card] = 0.5
                one_hot = self.flatten(state)
                evaluation = self.graph.evaluate(one_hot)
                if evaluation < best_w_disc:
                    (best_w_disc, to_discard_w_disc) = (evaluation, card)

                # Put it back for the next loop
                state[card] = 1
        if to_discard_w_disc is None:
            raise ValueError('Somehow, we never accepted a discard in discard_strategy()')
        return [best_w_disc, to_discard_w_disc]

    def draw_strategies(self, deck):

        state = deck.game_state.copy()
        cur_hand = self.hand[:]
        drawable = [k for k in state if state[k] == 0]

        # Dict with keys : value pairs drawable card : (best resulting discard, evaluation of play)
        strats_and_evals = {}

        if self.is_player_1:

            for card in drawable:
                state[card] = -1
                cur_hand.append(card)
                (best_discard, best_evaluation) = (None, inf)
                for disc in cur_hand:
                    state[disc] = -0.5
                    one_hot = self.flatten(state)
                    evaluation = self.graph.evaluate(one_hot)
                    if evaluation == inf:
                        raise ValueError('The neural-net is returning infinite values (in strats)')
                    if evaluation < best_evaluation:
                        (best_discard, best_evaluation) = (disc, evaluation)
                    state[disc] = -1
                strats_and_evals[card] = (best_discard, best_evaluation)
                state[card] = 0
                cur_hand.remove(card)
                if best_discard is None:
                    raise ValueError('Somehow, we never accepted a discard in strats_and_evals()')

        else:

            for card in drawable:
                state[card] = 1
                cur_hand.append(card)
                (best_discard, best_evaluation) = (None, inf)
                for disc in cur_hand:
                    state[disc] = 0.5
                    one_hot = self.flatten(state)
                    evaluation = self.graph.evaluate(one_hot)
                    if evaluation < best_evaluation:
                        (best_discard, best_evaluation) = (disc, evaluation)
                    state[disc] = 1
                strats_and_evals[card] = (best_discard, best_evaluation)
                state[card] = 0
                cur_hand.remove(card)

        return strats_and_evals

    # !Large issue right now: we are giving this player information he should not have!!
    # !Need to ensure that the player only 'sees' cards in the other player's hand if he
    # !has explicitly watched the other player pick them up.
    def play(self, deck):

        # Query the NN: what is the best score I can expect from picking up the discard
        # and then discarding something else? Work with a copy of the game state so we can
        # modify it freely.

        one_hot = self.flatten(deck.game_state)
        best_now = self.graph.evaluate(one_hot, training=self.training)

        [best_w_disc, to_discard_w_disc] = self.discard_strategy(deck)
        strats_and_evals = self.draw_strategies(deck)
        draw_scores = [strats_and_evals[strat][1] for strat in strats_and_evals]
        mean_draw = np.mean(draw_scores)
        best_drawing = max(mean_draw, best_w_disc)

        # If no greener pastures, knock now:
        if best_now > best_drawing:
            if min_loose_points(self.hand)[0] < 10:
                # Have to write best_meld static method
                optimal_meld = min_loose_points(self.hand)[1]
                deck.knock = True
                deck.ending_player_melds = optimal_meld
                return

        if mean_draw < best_w_disc:

            # Draw from the discard pile and discard to_discard_w_disc
            self.draw_discard(deck)
            self.discard(to_discard_w_disc, deck)

        else:

            # Draw from the deck and discard the associated card
            self.draw_deck(deck)
            drawn = self.hand[-1]
            to_discard = strats_and_evals[drawn][0]
            self.discard(to_discard, deck)


class Game:

    def __init__(self, deck, player_1, player_2):

        self.player_1 = player_1
        self.player_2 = player_2
        self.deck = deck

        self.deck.deal(self.player_1, self.player_2)

    def has_gin(self, player):

        hand = player.hand

        if min_loose_points(hand)[0] == 0:
            return True
        else:
            return False

    def score_game(self, ending_player, other_player):

        # If the ending player has gin, the calculation is easy.
        if self.has_gin(ending_player):
            loose = min_loose_points(other_player.hand)[0]
            return 25 + loose

        # Else, count the deadwood of ending_player
        ep_hand = ending_player.hand[:]
        for meld in self.deck.ending_player_melds:
            for card in meld:
                ep_hand.remove(card)
        ep_deadwood = sum([value(card) for card in ep_hand])

        # Find all the cards which can be laid off:
        droppable = layoffable(self.deck.ending_player_melds)
        op_hand = other_player.hand[:]
        op_must_meld = [card for card in op_hand if card not in droppable]
        op_deadwood = inf
        for S in powerset(droppable):
            to_meld = op_must_meld[:]
            to_meld.extend(S)
            loose_points = min_loose_points(to_meld)[0]
            if loose_points < op_deadwood:
                op_deadwood = loose_points

        diff = op_deadwood - ep_deadwood
        if diff >= 0:
            return diff
        else:
            return 25 - diff

    def play_game(self, trace=False):

        while len(self.deck.deck_order) > 2:

            turn = self.deck.turn % 2

            if turn == 1:
                self.player_1.play(self.deck)
                if trace:
                    cur_dw = min_loose_points(self.player_1.hand)[0]
                    print('At the end of this turn, Player 1 has, with '
                          + str(cur_dw) + ' loose points: ')
                    print(self.player_1.hand)
                    if self.deck.knock:
                        print(self.deck.ending_player_melds)
            else:
                self.player_2.play(self.deck)
                if trace:
                    cur_dw = min_loose_points(self.player_2.hand)[0]
                    print('At the end of this turn, Player 2 has, with '
                          + str(cur_dw) + ' loose points: ')
                    print(self.player_2.hand)
                    print(self.deck.knock)
                    if self.deck.knock:
                        print(self.deck.ending_player_melds)

            self.deck.turn = self.deck.turn + 1

            # Check to see if the player has knocked, or if
            # the player has gin.
            if self.deck.knock:
                if turn == 1:
                    pts = self.score_game(self.player_1, self.player_2)
                    if pts >= 0:
                        print('Player 1 wins ' + str(pts) + ' points')
                    else:
                        print('Player 2 wins ' + str(pts) + ' points')
                else:
                    pts = self.score_game(self.player_2, self.player_1)
                    if pts >= 0:
                        print('Player 2 wins ' + str(pts) + ' points')
                    else:
                        print('Player 1 wins ' + str(pts) + ' points')

                return

        # You've gone through the entire deck, so you're done.
        print('No score for this game!')


class TrainingGame(Game):
    """
    A game in which the first player is a neural-net player. At the end of the game,
    the NN player learns by applying stored gradients, then clears its memory of gradients for the next
    run.
    """

    def play_game(self, trace=False):

        while len(self.deck.deck_order) > 2:

            turn = self.deck.turn % 2

            if turn == 1:
                self.player_1.play(self.deck)

                if trace:
                    cur_dw = min_loose_points(self.player_1.hand)[0]
                    print('At the end of this turn, Player 1 has, with '
                          + str(cur_dw) + ' loose points: ')
                    print(self.player_1.hand)
                    if self.deck.knock:
                        print(self.deck.ending_player_melds)
            else:
                self.player_2.play(self.deck)
                if trace:
                    cur_dw = min_loose_points(self.player_2.hand)[0]
                    print('At the end of this turn, Player 2 has, with '
                          + str(cur_dw) + ' loose points: ')
                    print(self.player_2.hand)
                    print(self.deck.knock)
                    if self.deck.knock:
                        print(self.deck.ending_player_melds)

            self.deck.turn = self.deck.turn + 1

            # Check to see if the player has knocked, or if
            # the player has gin.
            if self.deck.knock:
                if turn == 1:
                    pts = self.score_game(self.player_1, self.player_2)
                    self.player_1.graph.end_game(pts)
                    if pts >= 0:
                        print('Player 1 wins ' + str(pts) + ' points')
                    else:
                        print('Player 2 wins ' + str(pts) + ' points')
                else:
                    pts = self.score_game(self.player_2, self.player_1)
                    self.player_1.graph.end_game(-pts)
                    if pts >= 0:
                        print('Player 2 wins ' + str(pts) + ' points')
                    else:
                        print('Player 1 wins ' + str(pts) + ' points')

                return

        print('Exhausted deck, neither player knocked')

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
    for i in range(n):
        deck = Deck()
        player_1 = NNPlayer()
        player_2 = NNPlayer()
        game = Game(deck, player_1, player_2)
    t_2 = time()
    print('Took ' + str(t_2-t_1))

time_check(5)
time_check(10)
# time_check(50)
'''

