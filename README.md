# gin

A slow-burn project inspired by Arthur Samuel's self-taught checkers player and Gerard Tesauro's TD-Gammon, a self-taught
backgammon player. The goal is to train a neural net to gain proficiency playing the card game gin using the temporal-difference algorithm 
(as explicated [here](https://web.stanford.edu/group/pdplab/pdphandbook/handbookch10.html)).

## Rules of gin

Amongst the many variants of gin, we focus on a simple one-game variant. The standard 52-card deck is used.

Throughout the game, players maintain a hand of 10 cards. The eventual goal is to unify all cards into 'runs' (sequences within a 
suit of consecutive numbers, with a length of at least 3 cards) or 'melds' (groups of cards all of the same number, of at least 3 cards),
minimizing the value of the cards not participating in some run or meld, where the value of a card 1-10 is it's face value; J, Q, K all carry
value of 10 as well. (Note: a card can only participate in 1 run/meld at a time)

Once a player has fewer than 10 points unmatched, (such cards are called 'deadwood', in the language of the game), he/she has the option
to 'knock', and end the game. There are now two different scoring rules based on whether or not the knocking player has 0 or more points 
of deadwood.

If the knocking player has more than 0 points of deadwood, they lay out all their melds on the table. The opposing player then has the chance
to lay down all of his/her melds on the table in any way they see fit, with the ability to lay off any of their deadwood onto the runs/melds
of the knocking player if those cards would extend them. After this is done, the difference between the remaining deadwood in the 
opponents' hand and the knocking player's hand is counted. If it is greater than 0, the knocking player recieves the difference in points.
If it is less than 0, the opposing player 'cuts' the knock, and recieves 25 points plus the difference in points. If the difference is 0,
nothing happens and game is a 'scratch'.

If instead the knocking player has 0 points of deadwood, they are said to have 'gin', and the opposing player is given no chance to
lay off any of their deadwood onto the gin hand. The player who went gin gains 25 points plus the value of the deadwood carried by the
opposing player.

## Current layout

* utils.py - Implementation of essential scoring/card type mechanics, as well as a Deck class.
* players.py - Various player classes, including a simple greedy player and the neural net player.
* game.py - A class for building games (currently missing a console-interactive game)
* simplenn.py - The TensorFlow graph architecture that sits inside the NNPlayer class as it's engine.
* dojo.py - 'I know kung fu' - 'Show me' - a command-line utility for training a simple NNPlayer over various games.
* main.py - A yet-not-implemented command line interface supporting training various players, assessing statistics.

## Planned improvements

* Wrapping the TD-lambda algorithm into a single TensorFlow Operation for use across variable neural-net architectures
* Experimenting with different embeddings of the game state.




