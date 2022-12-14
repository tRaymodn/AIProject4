import csv
import datetime

import numpy as np
from tabulate import tabulate

date_format = "%Y-%m-%d"

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.neural_network import MLPClassifier
from scipy.stats import pearsonr

# Read in data from csv
data = []
with open("fullDataset.csv") as file:
    reader = csv.reader(file)
    for row in reader:
        data.append(row)

# Returns a list of tuples containing a date, a team that played a game on that date, and the correct label for what team won
def getAllGames():
    games = []
    for row in data[1:]:
        gameInstance = [row[68], row[29], row[30]]
        flag = 0
        for game in games:
            if gameInstance == game or game == [row[68], row[30], row[29]]:  # Catch repeat games with different home and away teams
                flag = 1
        if flag == 0:
            games.append(gameInstance)
    return games

# Accepts a three-letter abbreviation of a team and a string date of the form YYYY-mm-dd
# Returns cumulative statistics(hyperparameters) from all games the given team has played up until the given date
def getTeamData(team, date):
    current_date = datetime.datetime.strptime(date, date_format)
    teamData = []
    for row in data[1:]:
        game_date = datetime.datetime.strptime(row[68], date_format)
        cutoff_date = datetime.datetime(game_date.year - 1, game_date.month, game_date.day)
        if row[4] == team and current_date > game_date > cutoff_date:
            if len(teamData) < 1:
                for i in range(len(row)):
                    if 5 <= i <= 28 or 30 < i <= 54 or 57 <= i <= 58:
                        teamData.append(float(row[i]))
            else:
                dc = 0
                for i in range(len(row)):
                    if 5 <= i <= 28 or 30 < i <= 54:
                        teamData[dc] += float(row[i])
                        dc += 1
    return teamData

# Returns a list of all teams in the NFL
def getTeams():
    teams = []
    for row in data:
        if row[4] not in teams and row[4] != 'team':
            teams.append(row[4])
        if len(teams) > 31:
            break
    return teams
def main():
    # Populate dataset with values to be input into neural network
    games = getAllGames()
    gamesplit = games[20:285]
    dataset = []
    for game in gamesplit:
        t1 = getTeamData(game[1], game[0])
        t2 = getTeamData(game[2], game[0])
        t2.pop()
        t2.pop()
        combined = t1
        for param in t2:
            combined.append(param)
        dataset.append(combined)

    split = train_test_split(dataset, test_size=0.3, random_state=42, shuffle=True)  # split[1]=testing, split[0]=training
    print("Length of training data : ", len(split[0]))
    labels = []
    for row in split[0]:
        if row[48] > row[49]:
            labels.append(0)
        else:
            labels.append(1)

    print("Length of testing data : ", len(split[1]))
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(6, 3), random_state=1, max_iter=2500)
    clf.fit(split[0], labels)

    #  Hyperparameters to be tuned in GridSearchCV
    randomState = np.array([1, 5, 10])
    tol = np.array([0.0001, 0.00001, 0.01])
    alpha = np.array([0.0001, 0.001, 0.01, 0.1, 1])
    grid = GridSearchCV(estimator=clf, param_grid={'tol': tol, 'alpha': alpha, 'random_state': randomState})
    grid.fit(split[0], labels)
    # summarize the results of the grid search
    print("Best grid score: ", grid.best_score_)
    print("Best estimator for random state: ", grid.best_estimator_.random_state)
    print("Best estimator for tolerance: ", grid.best_estimator_.tol)
    print("Best estimator for alpha: ", grid.best_estimator_.alpha)

    #  Create a neural network with the optimal parameters from GridSearchCV
    bestEstimate = MLPClassifier(solver='lbfgs', random_state=grid.best_estimator_.random_state,
                                 tol=grid.best_estimator_.tol,
                                 alpha=grid.best_estimator_.alpha,
                                 hidden_layer_sizes=(6, 3),
                                 max_iter=2000)
    scores = cross_val_score(bestEstimate, split[0], labels, cv=5)
    print("5-fold cross validation scores on training data: ", scores)
    f1 = sum(scores) / len(scores)
    print("Average cross validation score on training data: ", f1)

    #  Record the names of all of the statistics' correlations to be calculated
    stat_names = []
    for i in range(len(data[0])):
        if 5 <= i <= 28 or 30 < i <= 54 or 57 <= i <= 58:
            stat_names.append(data[0][i])
    #  print(stat_names)

    #  Creates a table of tuples - ['statName', 'pearsonCorrelation', 'pValue']
    correlation_table = []
    for i in range(0, 50):
        correlation_row = []
        for row in split[0]:
            correlation_row.append(row[i])
        corr = pearsonr(correlation_row, labels)  # Measures the correlation between each param and labels
        table_input = [stat_names[i], corr.statistic, corr.pvalue]
        correlation_table.append(table_input)
    #  Print all the parameters and their corresponding pvals and correlation statistics
    print(tabulate(correlation_table, headers=["Statistic", "Correlation", "P_val"]))





        
if __name__ == "__main__":
      main()