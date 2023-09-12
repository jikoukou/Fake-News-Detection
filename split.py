import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import *
import numpy as np


def Return_values():
    nltk.download('stopwords')
    nltk.download('punkt')

    dataset = pd.read_csv("NewDataset.csv")

    # Delete rows that were not found (Seite nicht gefunden)
    dataset = dataset.loc[dataset["Title"] != 'Seite nicht gefunden']

    # Delete forbidden websites
    dataset = dataset.loc[dataset["Title"] != '403 Forbidden']
    # dataset = dataset.loc[dataset["Ratio_of_Fake_Statements"] != 9]
    # Extract the titles only
    x = dataset.iloc[:, 1].values

    # Sentences after the removal of stop words
    filtered_x = []
    ps = SnowballStemmer("german")

    for i in x:
        text_tokens = word_tokenize(i)
        tokens_without_sw = [
            word for word in text_tokens if not word in stopwords.words()]
        for j in range(len(tokens_without_sw)):
            tokens_without_sw[j] = ps.stem(tokens_without_sw[j])
        filtered_sentence = (" ").join(tokens_without_sw)

        filtered_x.append(filtered_sentence)

    punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    max_length = 0
    for i in range(len(filtered_x)):

        for ele in filtered_x[i]:
            if ele in punc:
                filtered_x[i] = filtered_x[i].replace(ele, "")
        if (len(filtered_x[i].split()) > max_length):
            max_length = len(filtered_x[i].split())
    filtered_x = np.asarray(filtered_x)

    t = Tokenizer()
    t.fit_on_texts(filtered_x)
    vocab_size = len(t.word_index) + 1
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(filtered_x)
    # pad documents to a max lengt
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length)

    wanted_columns = pd.get_dummies(dataset, columns=['Overall_Rating'])
    y = (wanted_columns).iloc[:, 3:len(
        (pd.get_dummies(dataset, columns=['Overall_Rating'])).columns)].values

    new_y = np.zeros((len(y), 2))
    for i in range(len(y)):
        for j in range(len(y[0])):
            if (y[i][j] == 1):
                if (j < (len(y[0]))/2-1):
                    new_y[i][0] = 0.
                    new_y[i][1] = 1.
                else:
                    new_y[i][0] = 1.
                    new_y[i][1] = 0.
    y = new_y
    x_train, x_test, y_train, y_test = train_test_split(
        padded_docs, y, test_size=0.2, random_state=0)

    vector_space = 64

    return_value = []
    return_value.append(x_train)
    return_value.append(x_test)
    return_value.append(y_train)
    return_value.append(y_test)
    return_value.append(vocab_size)
    return_value.append(vector_space)
    return_value.append(max_length)
    return return_value
