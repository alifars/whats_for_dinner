import json
import numpy as np
from numpy import argmax
import pandas as pd
# models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# keras imports
from keras.layers import Dense
from keras.models import Sequential



jtrain_file = open("train.json")
jtest_file = open("test.json")

train_data = json.load(jtrain_file)
test_data = json.load(jtest_file)

labels = [doc['cuisine'] for doc in train_data]




train_features = [','.join(doc['ingredients']) for doc in train_data]
train_df = pd.DataFrame()

train_df['features'] = train_features
train_df['labels'] = labels

test_features = [','.join(doc['ingredients']) for doc in test_data]
test_ids = [doc['id'] for doc in test_data]
test_df = pd.DataFrame()
test_df['features'] = test_features
test_df['ids'] = test_ids
#print(test_df)

features = train_df['features']

#vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','), max_features= 4000)
vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split(','), max_features= 4000)
encoded_train_feats = vectorizer.fit_transform(train_features)
encoded_test_feats = vectorizer.fit_transform(test_features)
encode_labels = pd.get_dummies(train_df['labels'])

labels_list = encode_labels.columns.values
print(len(labels_list))
i = 0
labels_dict = dict()
for label in labels_list:
    labels_dict[i] = label
    i +=1

print(labels_dict)

def mlp_model():

    model = Sequential()
    model.add(Dense(32, input_shape=(4000,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(20, activation='softmax'))
    model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# train


model = mlp_model()
model.fit(encoded_train_feats, encode_labels, epochs=1, batch_size=10)
int_preds = model.predict(encoded_test_feats)
preds = []
for pred in int_preds:
    max = np.argmax(pred)
    preds.append(labels_dict[max])

test_ids = pd.Series(data=test_ids)
pred_df = pd.DataFrame({'id': test_ids.values, 'cuisine': preds}, index=None)

pred_df.to_csv("submission.csv", index=None)
