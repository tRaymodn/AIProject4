import tensorflow as tf
import keras
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
from main import getTeamData, getAllGames

date_format = "%Y-%m-%d"


def normalize_data(data):
    pass
    

def make_model():
    # Populate dataset with sequences of teams' game stat sums to be input into neural network
    games = getAllGames()
    games = train_test_split(games, test_size=0.2, random_state=40, shuffle=False)
    teams_seen = []  # List of teams that have been reached in the loop
    sequence_labels = []  # List of true labels for each data point
    game_sequences = []  # Nested list of data for each team for every game they've played
    for game in games[0][:200]:
        if game[1] not in teams_seen and len(game[1]) == 3:  # If the team hasn't been seen yet, add it in the seen teams and game sequences
            teams_seen.append(game[1])
            game_sequences.append(getTeamData(game[1], game[0]))
            sequence_labels.append(game[3])
        elif len(game[1]) == 3:  # Adds game values if team has been seen already
            game_sequences[teams_seen.index(game[1])].append(getTeamData(game[1], game[0]))
            sequence_labels.append(game[3])

        if game[2] not in teams_seen and len(game[2]) == 3:
            teams_seen.append(game[2])
            game_sequences.append(getTeamData(game[2], game[0]))
            sequence_labels.append(game[3])
        elif len(game[2]) == 3:
            game_sequences[teams_seen.index(game[2])].append(getTeamData(game[2], game[0]))
            sequence_labels.append(game[3])
    print("Shape of game_sequences: ", np.shape(game_sequences))

    seq_num = 0
    labs = []
    arr = []
    """for seq in game_sequences:  # Runs through the 32 sequences of games for each team
        seq.pop(0)  # The first value in the sequence will always be empty because it is the first game the team played
        print("Sequence shape: ", np.shape(seq))
        for i in range(0, len(seq)):  # Loop through game values for a certain team
            print("game shape: ", np.shape(seq[i]))
            if i % 4 == 0:
                arr.append(seq[i])
                print("first game: ", seq[i])
            elif i % 4 == 1:
                # print("Shape of arr[len(arr) - 1]: ", np.shape(arr[len(arr) - 1]), "\ngame shape: ", np.shape(seq[i]))
                arr[len(arr) - 1] = np.stack((arr[len(arr) - 1], seq[i]))
            else:
                arr[len(arr) - 1] = np.concatenate((arr[len(arr) - 1], [seq[i]]))"""
    sequence_labels = []
    for seq in game_sequences:
        seq.pop(0)
        print("length of sequence: ", len(seq))
        print("seq: ", seq)
        i = 0
        sequence_clipped = []  # Array where the short sequences will be stored
        labels = []
        while i < 8:  # Creating two sequences of four games each from each team's full sequence of games
            if i == 4:
                arr.append(np.array(sequence_clipped))  # Add short sequence to the final array
                sequence_clipped = []  # Empty the array containing the short sequence
                if seq[i][48] > seq[i][49]:  # Add a 0 to the labels if the away team wins and a 1 otherwise
                    sequence_labels.append([0])
                else:
                    sequence_labels.append([1])
            sequence_clipped.append(np.array(seq[i]))
            i += 1
        arr.append(np.array(sequence_clipped))
        if seq[i-1][48] > seq[i-1][49]:
            sequence_labels.append([0])
        else:
            sequence_labels.append([1])
    print("Shape of arr: ", np.shape(arr))
    print("arr[0][0]: ", arr[0][0])
    labs = np.asarray(sequence_labels)
    print(labs)
    print("Shape of labs is: ", np.shape(labs))
    labs.reshape((64, 1, 1))
    print("labs reshaped is: ", np.shape(labs))
    arr = tf.stack(arr)

    print("label shape: ", np.shape(sequence_labels))
    model = Sequential()
    model.add(LSTM(20, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(20, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5)

    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'mse'])

    print(np.shape(arr))
    model.fit(arr, labs, epochs=200)

make_model()
