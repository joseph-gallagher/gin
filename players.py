from utils import *
from simple_nn import SimpleGraph


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

    # A completely random play method; never knocks.
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

            # An alternate way of doing this to minimize calls to Session.run()
            a = []
            d = {}
            index = 0
            for discard in cur_hand:
                state[discard] = -0.5
                one_hot = self.flatten(state)
                a.append(one_hot)
                d[discard] = index
                index = index + 1
                # Put the discard back for the next round.
                state[discard] = -1

            a = np.array(a)
            evaluation = self.graph.evaluate(a)

            (best_w_disc, to_discard_w_disc) = min([(evaluation[d[discard]], discard) for discard in d],
                                                   key=lambda x : x[0])
            '''    
            for card in cur_hand:

                # See what it would be like if you discarded it:
                state[card] = -0.5
                one_hot = self.flatten(state)
                evaluation = self.graph.evaluate([one_hot])
                if evaluation == inf:
                    raise ValueError('The neural-net is returning infinite values (in discard_strategy)')
                if evaluation < best_w_disc:
                    (best_w_disc, to_discard_w_disc) = (evaluation, card)

                # Put it back for the next loop
                state[card] = -1
            '''
        else:

            state[top_discard] = 1
            cur_hand.append(top_discard)

            # An alternate way of doing this to minimize calls to Session.run()
            a = []
            d = {}
            index = 0
            for discard in cur_hand:
                state[discard] = -0.5
                one_hot = self.flatten(state)
                a.append(one_hot)
                d[discard] = index
                index = index + 1
                # Put the discard back for the next round.
                state[discard] = -1

            a = np.array(a)
            evaluation = self.graph.evaluate(a)

            (best_w_disc, to_discard_w_disc) = min([(evaluation[d[discard]], discard) for discard in d],
                                                   key=lambda x: x[0])

            '''
            for card in cur_hand:

                # See what it would be like if you discarded it:
                state[card] = 0.5
                one_hot = self.flatten(state)
                evaluation = self.graph.evaluate(one_hot)
                if evaluation < best_w_disc:
                    (best_w_disc, to_discard_w_disc) = (evaluation, card)

                # Put it back for the next loop
                state[card] = 1
            '''
        if to_discard_w_disc is None:
            raise ValueError('Somehow, we never accepted a discard in discard_strategy()')
        return [best_w_disc, to_discard_w_disc]

    def draw_strategies(self, deck):

        state = deck.game_state.copy()
        cur_hand = self.hand[:]
        drawable = [k for k in state if state[k] == 0]

        # Dict with key : value pairs drawable card : (best resulting discard, evaluation of play)
        strats_and_evals = {}

        if self.is_player_1:

            # An alternate attempt to avoid multiple calls to sess.run()
            a = []
            d = {}
            index = 0
            for draw in drawable:
                # It is now in your hand
                state[draw] = -1
                cur_hand.append(draw)
                d[draw] = {}
                for discard in cur_hand:
                    # And the discard has been removed
                    state[discard] = -0.5
                    one_hot = self.flatten(state)
                    a.append(one_hot)
                    d[draw][discard] = index
                    index = index + 1
                    # Replace the discard for the next loop
                    state[discard] = -1
                # Replace the drawn card for the next loop
                cur_hand.remove(draw)
                state[draw] = 0

            # Just one call to self.sess.run()
            a = np.array(a)
            evaluation = self.graph.evaluate(a)

            # Fill in strats_and_evals by looping again through possibilities:
            for card in d:
                plays_and_evals = [(discard, evaluation[d[card][discard]]) for discard in d[card]]
                (best_discard, eval_of_play) = min(plays_and_evals, key=lambda x: x[1])
                strats_and_evals[card] = (best_discard, eval_of_play)

            '''
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
            '''

        else:

            # An alternate attempt to avoid multiple calls to sess.run()
            a = []
            d = {}
            index = 0
            for draw in drawable:
                # It is now in your hand
                state[draw] = 1
                cur_hand.append(draw)
                d[draw] = {}
                for discard in cur_hand:
                    # And the discard has been removed
                    state[discard] = 0.5
                    one_hot = self.flatten(state)
                    a.append(one_hot)
                    d[draw][discard] = index
                    index = index + 1
                    # Replace the discard for the next loop
                    state[discard] = 1
                # Replace the drawn card for the next loop
                cur_hand.remove(draw)
                state[draw] = 0

            # Just one call to self.sess.run()
            evaluation = self.graph.evaluate(a)

            # Fill in strats_and_evals by looping again through possibilities:
            for card in d:
                plays_and_evals = [(discard, evaluation[d[card][discard]]) for discard in d[card]]
                (best_discard, eval_of_play) = min(plays_and_evals, key=lambda x: x[1])
                strats_and_evals[card] = (best_discard, eval_of_play)

            '''
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
            '''
        return strats_and_evals

    # !Large issue right now: we are giving this player information he should not have!!
    # !Need to ensure that the player only 'sees' cards in the other player's hand if he
    # !has explicitly watched the other player pick them up.
    def play(self, deck):

        # Query the NN: what is the best score I can expect from picking up the discard
        # and then discarding something else? Work with a copy of the game state so we can
        # modify it freely.

        one_hot = self.flatten(deck.game_state)
        best_now = self.graph.evaluate([one_hot], training=self.training)

        [best_w_disc, to_discard_w_disc] = self.discard_strategy(deck)
        strats_and_evals = self.draw_strategies(deck)
        draw_scores = [strats_and_evals[strat][1] for strat in strats_and_evals]
        mean_draw = np.mean(draw_scores)
        best_drawing = max(mean_draw, best_w_disc)

        # If no greener pastures, knock now:
        if best_now < best_drawing:
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

