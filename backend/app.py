from flask import Flask, request
import time
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def run_logistic_regression(df):
  X = df.drop(columns=['team1_win'])
  y = df['team1_win']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = LogisticRegression()
  model.fit(X_train, y_train)

  # evaluate model
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  return accuracy

# HOW TO RUN

# navigate to the backend folder
# $ cd backend

# activate the virtual environment
# $ source venv/bin/activate

# run the Flask app
# $ python -m flask run --reload (restarts server after changes)

app = Flask(__name__)

# helper functions

def compute_winners(scores):
    # empty result to hold results
    res = []
    for i in range(0, len(scores), 2):
        # if team 1 scores higher than team 2 score, team 1 wins & vice vrsa
        if scores[i] > scores[i+1]:
            res.append(1)
        else:
            res.append(0)
    return res

def process_predictions(pred_scores):
    return [int(max(0, score)) for score in pred_scores]

def compute_accuracy(pred_scores, actual_scores):
    # winner determined by model
    pred_winners = compute_winners(pred_scores)
    # actual winner of the game
    actual_winners = compute_winners(actual_scores)
    # computing accuracy
    return sum([1 if pred_winners[i] == actual_winners[i] else 0 for i in range(len(pred_winners))]) / len(pred_winners)

def load_df(binary, fields, start_year, end_year):
    if binary:
        df = pd.read_csv('assets/NFL_data_2013_2022_binary_True.csv')
    else:
        df = pd.read_csv('assets/NFL_data_2013_2022_binary_False.csv')
    
    # filter by years
    df = df[(df['year'] >= start_year) & (df['year'] <= end_year)]

    # filter by fields
    df = df[fields]

    return df

def drop_columns(df):
    # to avoid errors if we run this function multiple times
    if 'year' in df.columns:
        df.drop(columns=['year', 'week', 'postseason', 'team1', 'team2'], inplace=True)
    return df

# post request
@app.route('/accuracy', methods=['POST'])
def get_accuracy():
    # get data from request
    data = request.get_json()

    # get binary from data
    binary = bool(data['binary'])

    # get start_year from data
    start_year = int(data['start_year'])

    # get end_year from data
    end_year = int(data['end_year'])

    # get fields from data
    fields = list(data['fields'])
    if binary:
        fields.append('team1_win')
    else:
        fields.append('team1_score')

    # load data
    df = load_df(binary, fields, start_year, end_year)

    print(df.head())

    # drop columns
    df = drop_columns(df)

    # run logistic regression
    accuracy = run_logistic_regression(df)

    return str(accuracy)

@app.route('/time')
def get_current_time():
    return {'time': time.time()}

if __name__ == "__main__":
    app.run(port=8000, debug=True)