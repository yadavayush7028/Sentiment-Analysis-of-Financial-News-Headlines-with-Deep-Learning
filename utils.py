import numpy as np
import os

from keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def f1(y_true, y_pred):
    
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def vocab_size(data):
    results = set()
    data['processed_text'].str.lower().str.split().apply(results.update)
    return len(results)
    
def max_len(data):
    maxlen = data['processed_text'].str.split().str.len().max()
    return maxlen

def tokenizer(data):

    max_features = vocab_size(data)
    maxlen = max_len(data)
    tokenizer = Tokenizer(num_words=max_features, split=' ')
    tokenizer.fit_on_texts(data['processed_text'].values)
    X = tokenizer.texts_to_sequences(data['processed_text'].values)
    X = pad_sequences(X, maxlen = maxlen)
    y = data['lbl'].values

    return X,y    
