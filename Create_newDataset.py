# Some imports needed
import pandas as pd
import lxml.html
import matplotlib.pyplot as plt
import tensorflow as tf
from django.core.validators import URLValidator
from django.core.exceptions import ValidationError
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Dense, LSTM, Flatten, TimeDistributed
from tensorflow.keras.metrics import RootMeanSquaredError as rmse
from tensorflow.keras.optimizers import Adam
import numpy as np
import tensorflow as tf


def Create_newDataset():

    # converting json file (dataset) to csv

    fakenews_info = pd.read_json('GermanFakeNC.json')
    fakenews_info.to_csv('GermanFakeNC.csv', index=False)
    print(fakenews_info)
    df_dataset = pd.read_csv("GermanFakeNC.csv")

    df = df_dataset.drop(["Date", "False_Statement_1_Location", "False_Statement_1_Index", "False_Statement_2_Location",
                         "False_Statement_2_Index", "False_Statement_3_Location", "False_Statement_3_Index"], axis=1)

    print(df)
    df.describe()

    overall = df_dataset["Overall_Rating"]
    overall.to_numpy

    urls = df_dataset["URL"]
    urls.to_numpy()
    counter = 0

    ratio_Fake = df_dataset["Ratio_of_Fake_Statements"]
    ratio_Fake.to_numpy()

    x = []
    y = []
    z = []

    for i in urls:
        val = URLValidator()
        try:
            val(i)
            reqs = requests.get(i)
            soup = BeautifulSoup(reqs.text, 'html.parser')

            for title in soup.find_all('title'):
                x.append(title.get_text())
                y.append(overall[counter])
                z.append(ratio_Fake[counter])
        except:
            print(counter)

        counter = counter+1


Create_newDataset()
