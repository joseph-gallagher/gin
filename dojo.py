# A very simple way to simulate a few games of 'gyeah' versus Greedy.
from utils import *
from game import *
from players import *
import os
import matplotlib.pyplot as plt

current_dir = os.getcwd()
hero = 'gyeah'
player_1_path = os.path.join(current_dir, hero)
player_1_path = os.path.join(player_1_path, hero)

player_1 = NNPlayer(training=True)
plt.hist(player_1.graph.sess.run(player_1.graph.w1))
plt.show()

NUM_GAMES = 50
LEARN_EVERY = 10

for i in range(NUM_GAMES):
    deck = Deck()
    player_2 = Greedy_Player()
    karate = TrainingGame(deck, player_1, player_2)
    karate.play_game()
    if i > 0 and i % LEARN_EVERY == 0:
        player_1.graph.learn()

    # Re-initialize all of player_1's non-graph fields:
    player_1.hand = []

# Generate a histogram of the weights values to observe blowup.
plt.hist(player_1.graph.sess.run(player_1.graph.w1))
plt.show()

player_1.graph.save_state(save_path=player_1_path)
print('And now for the comparison:')
for i in range(5):
    deck = Deck()
    goofy = Player()
    smart = Greedy_Player()
    karate = Game(deck, goofy, smart)
    karate.play_game()