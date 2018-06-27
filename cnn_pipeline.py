import json
import numpy as np
import pandas as pd
# models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# keras imports
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Conv1D, Embedding, MaxPooling1D, Flatten
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

def read_files():
    # jtrain_file = open("../input/train.json")
    # jtest_file = open("../input/test.json")

    jtrain_file = open("train.json")
    jtest_file = open("test.json")

    train_data = json.load(jtrain_file)
    test_data = json.load(jtest_file)
    return train_data, test_data


def encode_labels(train_data):
    labels = [doc['cuisine'] for doc in train_data]
    encoded_labels = pd.get_dummies(labels)
    labels_list = list(encoded_labels)
    i = 0
    labels_dict = dict()
    for label in labels_list:
        labels_dict[i] = label
        i += 1

    return encoded_labels, labels_dict


def get_features(data):
    text_features = [','.join(doc['ingredients']) for doc in data]
    return text_features


def create_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    return tokenizer


def encode_features(tokenizer, max_length, texts):
    encoded = tokenizer.texts_to_sequences(texts)
    # encoded = tokenizer.texts_to_matrix(texts, 'tfidf')
    padded = pad_sequences(encoded, maxlen=max_length, padding='post')
    return padded


def define_model(vocab_size, max_length):
    # A CNN model for text classification
    model = Sequential()
    model.add(Embedding(vocab_size, 100, input_length=max_length))
    model.add(Conv1D(filters=32, kernel_size=8, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(20, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    # plot_model(model, show_shapes=True)
    return model

def get_test_ids(test_data):
    test_ids = [doc['id'] for doc in test_data]
    return  test_ids


def run():
    train_data, test_data = read_files()
    test_ids = get_test_ids(test_data)

    encoded_labels, labels_dict = encode_labels(train_data)
    train_features_text = get_features(train_data)
    test_features_text = get_features(test_data)
    tokenizer = create_tokenizer(train_features_text)
    max_length = np.max([len(text) for text in train_features_text])
    print(max_length)
    train_features = encode_features(tokenizer, max_length=max_length, texts=train_features_text)
    test_features = encode_features(tokenizer, max_length=max_length, texts=test_features_text)
    print(train_features)
    print(test_features)
    vocab_size = len(tokenizer.word_index) + 1
    model = define_model(vocab_size=vocab_size, max_length=max_length)
    model.fit(train_features, encoded_labels, epochs= 3, batch_size= 500)
    preds_probs = model.predict(test_features)

    preds = []
    for pred in preds_probs:
        max = np.argmax(pred)
        preds.append(labels_dict[max])

    test_ids = pd.Series(data=test_ids)
    pred_df = pd.DataFrame({'id': test_ids.values, 'cuisine': preds}, index=None)

    pred_df.to_csv("submission.csv", index=None)


run()
