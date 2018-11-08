from utils import *
from players import *


class Game:

    def __init__(self, deck, player_1, player_2):

        self.player_1 = player_1
        self.player_2 = player_2
        self.deck = deck

        self.deck.deal(self.player_1, self.player_2)

    def score_game(self, ending_player, other_player):

        # If the ending player has gin, the calculation is easy.
        ep_deadwood = min_loose_points(ending_player.hand)[0]
        if ep_deadwood == 0:
            op_deadwood = min_loose_points(other_player.hand)[0]
            return 25 + op_deadwood

        # Find all the cards which can be laid off:
        droppable = layoffable(self.deck.ending_player_melds)
        op_must_meld = [card for card in other_player.hand if card not in droppable]
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
            return -25 + diff

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
        pts = 0
        self.player_1.graph.end_game(pts)
