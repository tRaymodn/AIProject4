import csv

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def main():
    data = []
    with open("nfl_pass_rush_receive_raw_data.csv") as file:
        reader = csv.reader(file)
        for row in reader:
            data.append(row)
    split = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)  #split[1]=testing, split[0]=training
    """clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=2000)
    clf.fit(trainingFloatList, y_true)"""
    def getTeam(id):
        teamlist = []
        for row in data:
            if row[0] == id:
                teamlist.append(row)
        
if __name__ == "__main__":
      main()