from flask import Flask, request
import time
import pandas as pd

# HOW TO RUN

# navigate to the backend folder
# $ cd backend

# activate the virtual environment
# $ source venv/bin/activate

# run the Flask app
# $ python -m flask run --reload (restarts server after changes)

app = Flask(__name__)

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


## Models

# LR

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

# RF

from sklearn.ensemble import RandomForestClassifier

def run_random_forest_binary(df):
  X = df.drop(columns=['team1_win'])
  y = df['team1_win']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = RandomForestClassifier()
  model.fit(X_train, y_train)

  # evaluate model
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  return accuracy

# NB

from sklearn.naive_bayes import GaussianNB

def run_naive_bayes(df):
  X = df.drop(columns=['team1_win'])
  y = df['team1_win']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = GaussianNB()
  model.fit(X_train, y_train)

  # evaluate model
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  return accuracy

# SVC

from sklearn.svm import SVC

def run_svm_binary(df):
  X = df.drop(columns=['team1_win'])
  y = df['team1_win']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = SVC()
  model.fit(X_train, y_train)

  # evaluate model
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  return accuracy

# NN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

def run_nn_binary(df):
  X = df.drop(columns=['team1_win'])
  y = df['team1_win']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = Sequential()
  model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
  model.add(Dense(units=32, activation='relu'))
  model.add(Dense(units=1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

  # early stopping to prevent overfitting
  early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.15, callbacks=[early_stopping], verbose=0)

  _, accuracy = model.evaluate(X_test, y_test)

  return accuracy

# Non-binary

# LR

from sklearn.linear_model import LinearRegression

def run_linear_regression(df):
  X = df.drop(columns=['team1_score'])
  y = df['team1_score']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # evaluate model
  predictions = process_predictions(model.predict(X_test).flatten())
  accuracy = compute_accuracy(list(predictions), list(y_test))

  return accuracy

# RF

from sklearn.ensemble import RandomForestRegressor

def run_random_forest_regression(df):
  X = df.drop(columns=['team1_score'])
  y = df['team1_score']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = RandomForestRegressor()
  model.fit(X_train, y_train)

  # evaluate model
  predictions = process_predictions(model.predict(X_test).flatten())
  accuracy = compute_accuracy(list(predictions), list(y_test))

  return accuracy

# SVR

from sklearn.svm import SVR

def run_svm_regression(df):
  X = df.drop(columns=['team1_score'])
  y = df['team1_score']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = SVR()
  model.fit(X_train, y_train)

  # evaluate model
  predictions = process_predictions(model.predict(X_test).flatten())
  accuracy = compute_accuracy(list(predictions), list(y_test))

  return accuracy

# NN

def run_nn(df):
  X = df.drop(columns=['team1_score'])
  y = df['team1_score']

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

  # scale data
  scaler = StandardScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)

  # train model
  model = Sequential()
  model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
  model.add(Dense(units=32, activation='relu'))
  model.add(Dense(units=1, activation='linear'))

  model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01), metrics=['mse'])

  # early stopping to prevent overfitting
  early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
  model.fit(X_train, y_train, epochs=50, batch_size=16, validation_split=0.15, callbacks=[early_stopping], verbose=0)

  predictions = process_predictions(model.predict(X_test).flatten())
  accuracy = compute_accuracy(list(predictions), list(y_test))

  return accuracy

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
    
    model = data['model']

    # load data
    df = load_df(binary, fields, start_year, end_year)

    # drop columns
    df = drop_columns(df)

    if model == 'LR' and binary:
        accuracy = run_logistic_regression(df)
    elif model == 'RF' and binary:
        accuracy = run_random_forest_binary(df)
    elif model == 'NB' and binary:
        accuracy = run_naive_bayes(df)
    elif model == 'SVC' and binary:
        accuracy = run_svm_binary(df)
    elif model == 'NN' and binary:
        accuracy = run_nn_binary(df)
    elif model == 'LR' and not binary:
        accuracy = run_linear_regression(df)
    elif model == 'RF' and not binary:
        accuracy = run_random_forest_regression(df)
    elif model == 'SVC' and not binary:
        accuracy = run_svm_regression(df)
    elif model == 'NN' and not binary:
        accuracy = run_nn(df)

    return str(accuracy)

if __name__ == "__main__":
    app.run(port=8000, debug=True)