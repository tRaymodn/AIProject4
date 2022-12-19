import tensorflow as tf
import keras
import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
from main import getTeamData, getAllGames

date_format = "%Y-%m-%d"
    
def normalize(data):
    return(data - np.min(data)) / (np.max(data) - np.min(data))

def make_model():
    # Populate dataset with sequences of teams' game stat sums to be input into neural network
    games = getAllGames()
    games = train_test_split(games, test_size=0.2, random_state=40, shuffle=False)
    teams_seen = []  # List of teams that have been reached in the loop
    game_sequences = []  # Nested list of data for each team for every game they've played
    for game in games[0][:800]:
        if game[1] not in teams_seen and len(game[1]) == 3:  # If the team hasn't been seen yet, add it in the seen teams and game sequences
            teams_seen.append(game[1])
            game_sequences.append(getTeamData(game[1], game[0]))
        elif len(game[1]) == 3:  # Adds game values if team has been seen already
            game_sequences[teams_seen.index(game[1])].append(getTeamData(game[1], game[0]))

        if game[2] not in teams_seen and len(game[2]) == 3:
            teams_seen.append(game[2])
            game_sequences.append(getTeamData(game[2], game[0]))
        elif len(game[2]) == 3:
            game_sequences[teams_seen.index(game[2])].append(getTeamData(game[2], game[0]))

    arr = []
    sequence_labels = []
    for seq in game_sequences:
        seq.pop(0)
        print("length of training sequence: ", len(seq))
        print("seq: ", seq)
        i = 0
        sequence_clipped = []  # Array where the short sequences will be stored
        while i < len(seq):  # Creating two sequences of four games each from each team's full sequence of games
            if i % 4 == 0 and i != 0:
                arr.append(np.array(sequence_clipped))  # Add short sequence to the final array
                sequence_clipped = []  # Empty the array containing the short sequence
                if seq[i][48] > seq[i][49]:  # Add a 0 to the labels if the away team wins and a 1 otherwise
                    sequence_labels.append([0])
                else:
                    sequence_labels.append([1])
            sequence_clipped.append(normalize(np.array(seq[i])))
            i += 1
        if len(sequence_clipped) == 4:
            arr.append(np.array(sequence_clipped))
            if seq[i - 1][48] > seq[i - 1][49]:
                sequence_labels.append([0])
            else:
                sequence_labels.append([1])

    print("Shape of arr: ", np.shape(arr))
    print("arr[0][0]: ", arr[0][0])
    labs = np.asarray(sequence_labels)
    print(labs)
    print("Shape of labs is: ", np.shape(labs))
    labs.reshape((len(sequence_labels), 1, 1))
    print("labs reshaped is: ", np.shape(labs))
    arr = tf.stack(arr)

    print("label shape: ", np.shape(sequence_labels))
    model = Sequential()
    model.add(BatchNormalization())
    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate=1e-3, weight_decay=1e-5)

    model.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy', 'mse'])

    print(np.shape(arr))
    model.fit(arr, labs, epochs=150)  # Fit the training data to the RNN model and run for 400 epochs

    # Put the testing data into game sequences by team
    testing_teams_seen = []
    testing_sequences = []
    for game in games[1][:200]:
        if game[1] not in testing_teams_seen and len(game[1]) == 3:  # If the team hasn't been seen yet, add it in the seen teams and game sequences
            testing_teams_seen.append(game[1])
            testing_sequences.append([getTeamData(game[1], game[0])])
        elif len(game[1]) == 3 and game[1] in testing_teams_seen:  # Adds game values if team has been seen already
            testing_sequences[testing_teams_seen.index(game[1])].append(getTeamData(game[1], game[0]))

        if game[2] not in testing_teams_seen and len(game[2]) == 3:
            testing_teams_seen.append(game[2])
            testing_sequences.append([getTeamData(game[2], game[0])])
        elif len(game[2]) == 3 and game[2] in testing_teams_seen:
            testing_sequences[testing_teams_seen.index(game[2])].append(getTeamData(game[2], game[0]))
    print("Shape of testing sequences: ", np.shape(testing_sequences))

    testing_arr = []
    testing_labels = []
    for seq in testing_sequences:
        print("length of testing sequence: ", len(seq))
        print("seq: ", seq)
        index = 0
        small_arr = []
        for index in range(2, 6):
            small_arr.append(normalize(np.array(seq[index])))
        testing_arr.append(np.array(small_arr))
        # testing_arr.append(np.array(seq[4:8]))
        print("Visitors score, home score for the real game:", seq[6][48], seq[6][49])
        if seq[6][48] > seq[6][49]:
            testing_labels.append([0])
        else:
            testing_labels.append([1])
        
    print("Shape of testing array: ", np.shape(testing_arr))

    y_pred = model.predict(np.array(testing_arr))
    
    print("Testing true labels: ", testing_labels)
    print("Testing predicted labels: ", y_pred)
    missed_vals = 0
    for index in range(0, len(y_pred)):
        if y_pred[index] > 0.5:
            y_pred[index] = 1
        else:
            y_pred[index] = 0
    points_mislabeled = 0
    for ind in range(0, len(testing_labels)):
        if testing_labels[ind] != y_pred[ind]:
            points_mislabeled += 1

    print("Number of games predicted incorrectly: ", points_mislabeled)
    print("Total points: ", len(testing_labels))


make_model()
