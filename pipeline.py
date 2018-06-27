
import json
import numpy as np
from numpy import argmax
import pandas as pd
# models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import  cross_val_score
from keras.utils import np_utils
from sklearn.preprocessing import OneHotEncoder

def read_json(fname):
    "reads from Json file and converts them to a pandas dataframe"
    data = None
    with open(fname) as f:
        data = json.load(f)
    # lower case
    labels = [doc['cuisine'] for doc in data]
    features = [','.join(doc['ingredients']) for doc in data]
    df = pd.DataFrame()
    df['features'] = features
    df['labels'] = labels
    return df

def process_train_data(df):
    # turn data into pandas dataframe to make preprocessing easier
    # vectorize the features
    features = df['features']
    print(features)
    vectorizer = CountVectorizer(tokenizer= lambda x: x.split(','))
    encoded_feats = vectorizer.fit_transform(features)
    encode_labels = pd.get_dummies(df['labels'])

    # one-hot encode label


    return encoded_feats, encode_labels




def train_model(features, labels):
    model = RandomForestClassifier(random_state=0, verbose=2)
    model.fit(features, labels)
    return model


def evaluate():

    data = read_json("train.json")
    encoded_feats, encode_labels = process_train_data(data)
    print(encoded_feats.shape)
    print(encode_labels.shape)

    model = train_model(encoded_feats, encode_labels)
    scores = cross_val_score(model, X=encoded_feats, y=encode_labels, cv=7)
    score = np.mean(scores)
    print(score)
def submit():
    train_data = read_json("train.json")
    test_data = read_json("test.json")

    encoded_feats, encode_labels = process_train_data(train_data)

    test_feats = process_test_data(test_data)

    model = train_model(encoded_feats, encode_labels)
    preds = model.predict(test_feats)
    print(preds)

submit()