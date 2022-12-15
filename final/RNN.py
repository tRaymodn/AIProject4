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
    teams_seen = []
    game_sequences = []
    sequence_labels = []
    for game in games:
        if game[1] not in teams_seen and len(game[1]) == 3:  # If the team hasn't been seen yet, add it in the seen teams and game sequences
            teams_seen.append(game[1])
            game_sequences.append(getTeamData(game[1], game[0]))
            sequence_labels.append(game[3])
        elif len(game[1]) == 3 and len(game_sequences[teams_seen.index(game[1])]) < 11:  # Only save record of sequences of ten games
            game_sequences[teams_seen.index(game[1])].append(getTeamData(game[1], game[0]))
        if game[2] not in teams_seen and len(game[2]) == 3:
            teams_seen.append(game[2])
            game_sequences.append(getTeamData(game[2], game[0]))
            sequence_labels.append(game[3])
        elif len(game[2]) == 3 and len(game_sequences[teams_seen.index(game[2])]) < 11:
            game_sequences[teams_seen.index(game[2])].append(getTeamData(game[2], game[0]))
    print("Length of game_sequences: ", len(game_sequences))

    arr = []
    for seq in game_sequences:
        if len(arr) < 1:
            seq.pop(0)
            arr.append(seq)
            print("first 12 length seq is: ", seq)
        elif len(arr) == 1:
            seq.pop(0)
            print("Shape or arr[0]: ", np.shape(arr[0]), "\nseq shape: ", np.shape(seq))
            print("seq :", seq)
            arr = np.stack((arr[0], seq))
        else:
            """if np.shape([seq]) == (1, 12, 50):
                print("You are stupid: ", [seq])
                seq.pop(11)"""
            seq.pop(0)
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

    print(np.shape(sequence_labels))
    print(np.shape(arr))
    model.fit(arr, np.asarray(sequence_labels, dtype='float'), epochs=200)

make_model()
