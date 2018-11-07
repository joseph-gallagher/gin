from utils import *
import os
import tensorflow as tf

# Find all NN players in the given directory.
current_dir = os.getcwd()

nn_players = []
# Assumes the folder containing a NNPlayer is that player's name:
for file in os.listdir(current_dir):
    try:
        contents = os.listdir(os.path.join(current_dir, file))
        if 'checkpoint' in contents:
            nn_players.append(file)
    except NotADirectoryError:
        continue

while True:

    # Give the list of choices to the user:
    print('Welcome to the TD-Gin experiment.')
    print('You have the following options:')
    print('1. Train/Build a neural-net player')
    print('2. Simulate a head-to-head matchup between players')
    print('3. Play a game against a model')
    print('4. Exit')
    choice = input('What do you choose? ')

    if choice == 1:

        print('The available neural-net players are: ')
        for player in nn_players:
            print(player)
        print('Please choose an option: ')
        print('1. Train existing player')
        print('2. Build new player')
        selection = input('What do you choose? ')
        if selection == 1:

            print('These are the available models: ')
            for player in nn_players:
                print(player)
            train_info = input('Enter a model name, then number of games to train: ')

            # Can make this far more robust. Consider names with spaces, training sessions taking more than
            # a few hours, etc....
            train_info = train_info.split()
            try:
                player_name = train_info[0]
                num_games = int(train_info[1])
                recovery_path = player_name + player_name + '-0'
                path = os.path.join(current_dir, recovery_path)

                player_1 = NNPlayer(path)
                player_2 = NNPlayer(path)

                # Have to write TrainingGame class.
                for i in range(num_games):
                    training_game = TrainingGame(player_1, player_2)
                    training_game.play()

            except ValueError:
                print('Must enter a number for games to train')

        if selection == 2:

            name = input("What is the new player's name? (No spaces)")
            save_path = os.path.join(current_dir, name)
            save_path = os.path.join(save_path, name)

            # Have to write 'save_to' method for NNPlayer
            current_player = NNPlayer()
            current_player.save_to(save_path)

    if choice == 2:

        print('The available neural-net players are: ')
        for player in nn_players:
            print(player)
        fighters = input('Select two players to play 10 games')
        fighters.split()
        player_1_path = os.path.join(current_dir, fighters[0])
        player_1_path = os.path.join(player_1_path, fighters[0])
        player_2_path = os.path.join(current_dir, fighters[1])
        player_2_path = os.path.join(player_2_path, fighters[1])
        player_1 = NNPlayer(is_player_1=True, path=player_1_path)
        player_2 = NNPlayer(is_player_1=False, path=player_2_path)

        # Play a handful of games, then save player_1's data at the end.
        for i in range(10):
            deck = Deck()
            training_game = TrainingGame(deck, player_1, player_2)
            TrainingGame.play_game(trace=False)

        player_1.graph.save_state(player_1_path)

    if choice == 3:

        print("Haven't implemented this yet!")

    if choice == 4:

        break

