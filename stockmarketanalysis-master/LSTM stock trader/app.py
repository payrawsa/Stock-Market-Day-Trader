from keras.models import load_model
import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.predict import DataLoader
from core.model import Model

model = load_model('saved_models/model.h5')

data = DataLoader(
    os.path.join('data', 'sp500.csv'),
    0,
    [
        "Close",
        "Volume"
    ]
)

x_test, y_test = data.get_test_data(
    seq_len=50,
    normalise=True
)

def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot( data, label='Prediction')
        plt.legend()
    plt.show()


prediction = model.predict(x_test)


print(len(prediction))

import pandas as pd
dataframe = pd.read_csv('data/Google.csv', nrows=200)
account = 10000
prices=dataframe.get('Close')
print(len(prices))
shares=0
for i in range(len(prediction)):
    if prediction[i] < 0:
        if shares!=0:
            account+=shares*prices.iloc[i]
            shares=0
            print("sold shares for: ", prices.iloc[i])
            print("account value is: ",account)
            print()

    else:
        if account >=prices[i]:
            number_of_shares= math.floor(account/prices.iloc[i])
            shares+=number_of_shares
            account -= number_of_shares*prices.iloc[i]
            print("Number of shares purchased: ", number_of_shares)
            print("purchase shares for: ", prices.iloc[i])
            print("account value is: ", account)
            print()
