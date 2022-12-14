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
    # Populate dataset with values to be input into neural network
    games = getAllGames()
    teams_seen = []
    game_sequences = []
    for game in games:
        if game[1] not in teams_seen:  # If the team hasn't been seen yet, add it in the seen teams and game sequences
            teams_seen.append(game[1])
            game_sequences.append(getTeamData(game[1], game[0]))
        elif len(game_sequences[teams_seen.index(game[1])]) < 11:  # Only save record of sequences of ten games
            game_sequences[teams_seen.index(game[1])].append(getTeamData(game[1], game[0]))

        if game[2] not in teams_seen:
            teams_seen.append(game[2])
            game_sequences.append(getTeamData(game[2], game[0]))
        elif len(game_sequences[teams_seen.index(game[1])]) < 11:
            game_sequences[teams_seen.index(game[2])].append(getTeamData(game[2], game[0]))

    """sequence_labels = []
    for seq in game_sequences:
        seq.pop(0)
        game_labels = []
        for game in seq:
            if len(game_labels) < 11:
                if game[48] > game[49]:
                    game_labels.append(0)
                else:
                    game_labels.append(1)
        if len(sequence_labels) < 1:
            sequence_labels.append(game_labels)
        elif len(sequence_labels) == 1:
            sequence_labels = np.stack((sequence_labels[0], game_labels))
        else:
            print(np.shape([game_labels]))
            if np.shape([game_labels]) == (1, 11):
                game_labels.pop(10)
            sequence_labels = np.concatenate((sequence_labels, [game_labels]))"""

    labs = []
    for seq in game_sequences:
        if seq[len(seq) - 1][48] > seq[len(seq) - 1][49]:
            labs.append(0)
        else:
            labs.append(1)
    arr = []
    for seq in game_sequences:
        if len(arr) < 1:
            arr.append(seq)
        elif len(arr) == 1:
            arr = np.stack((arr[0], seq))
        else:
            if np.shape([seq]) == (1, 12, 50):
                print("You are stupid: ", [seq])
                seq.pop(11)
            arr = np.concatenate((arr, [seq]))

        print(np.shape(seq))
    print(np.shape(arr))
    model = Sequential()
    model.add(LSTM(20, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(20, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5)

    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    print(np.shape(labs))
    print(np.shape(arr))
    model.fit(arr, np.asarray(labs, dtype='float'), epochs=200)

make_model()
