import numpy as np
import pandas as pd
import re

from data_clean import tknzr, stmr, lmtzr, process_df
from utils import f1

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, SimpleRNN
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical


def RNN_model(vocab_size, X, y, embed_dim, out_dim, num_epochs, batch_size):
    
    rnn = Sequential()
    rnn_out = out_dim
    rnn.add(Embedding(vocab_size, embed_dim, input_length = X.shape[1]))
    rnn.add(SpatialDropout1D(0.3))
    rnn.add(SimpleRNN(rnn_out, return_sequences=True, dropout=0.2))
    rnn.add(SimpleRNN(rnn_out, return_sequences=True, dropout=0.2))
    rnn.add(SimpleRNN(rnn_out, return_sequences=True, dropout=0.2))
    rnn.add(SimpleRNN(rnn_out, return_sequences=False, dropout=0.2))
    rnn.add(Dense(3, activation='softmax'))

    rnn.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = [f1])

    y = to_categorical(y, 3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    history = rnn.fit(X_train,
                        y_train,
                        epochs = num_epochs,
                        batch_size=batch_size,
                        verbose = 2,
                        validation_data=(X_test, y_test))
        
    HIST = pd.DataFrame(history.history)

    preds = rnn.predict(X_test)

    y_true = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(preds, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return rnn, HIST, acc, f1_macro