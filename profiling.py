# Profiling the play_game method for TrainingGame.

from utils import *
import os
import cProfile
import pstats

current_dir = os.getcwd()
hero = 'gyeah'
player_1_path = os.path.join(current_dir, hero)
player_1_path = os.path.join(player_1_path, hero)

player_1 = NNPlayer(training=False)

deck = Deck()
player_2 = Greedy_Player()
karate = TrainingGame(deck, player_1, player_2)

cProfile.run('karate.play_game()', 'gamestats')
p = pstats.Stats('gamestats')
p.strip_dirs()
p.sort_stats('time').print_stats(20)
