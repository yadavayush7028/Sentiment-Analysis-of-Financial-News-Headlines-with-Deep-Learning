import numpy as np
import pandas as pd
import re
import os

from data_clean import tknzr, stmr, lmtzr, process_df
from utils import f1

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical



def LSTM_model(vocab_size, X, y, embed_dim, out_dim, num_epochs, batch_size):
    
    lstm = Sequential()
    lstm_out = out_dim
    lstm.add(Embedding(vocab_size, embed_dim, input_length = X.shape[1]))
    lstm.add(SpatialDropout1D(0.4))
    lstm.add(LSTM(lstm_out, return_sequences=True, dropout=0.2))
    lstm.add(LSTM(lstm_out, return_sequences=True, dropout=0.2))
    lstm.add(LSTM(lstm_out, return_sequences=True, dropout=0.2))
    lstm.add(LSTM(lstm_out, dropout=0.2 ))
    lstm.add(Dense(3, activation='softmax'))

    lstm.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = [f1])

    y = to_categorical(y, 3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    history = lstm.fit(X_train,
                        y_train,
                        epochs = num_epochs,
                        batch_size=batch_size,
                        verbose = 2,
                        validation_data=(X_test, y_test))

    HIST = pd.DataFrame(history.history)

    preds = lstm.predict(X_test)

    y_true = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(preds, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return lstm, HIST, acc, f1_macro

# for func in [stmr, lmtzr]:

#     df = pd.read_csv('combined_data.csv')
#     labels = {'positive':2, 'neutral':1, 'negative':0}
#     df['lbl'] = df['sntmt'].map(lambda x:labels[x])
#     data = process_df(df, func, False)

#     max_features = 6000
#     tokenizer = Tokenizer(num_words=max_features, split=' ')
#     tokenizer.fit_on_texts(data['processed_text'].values)
#     X = tokenizer.texts_to_sequences(data['processed_text'].values)
#     X = pad_sequences(X)
#     y = data['lbl'].values
#     y = to_categorical(y, 3)

#     embed_dim = 256
#     lstm_out = 128

#     lstm = Sequential()

#     lstm.add(Embedding(max_features, embed_dim, input_length = X.shape[1]))
#     lstm.add(SpatialDropout1D(0.4))
#     lstm.add(LSTM(lstm_out, return_sequences=True, dropout=0.2))
#     lstm.add(LSTM(lstm_out, return_sequences=True, dropout=0.2))
#     lstm.add(LSTM(lstm_out, return_sequences=True, dropout=0.2))
#     lstm.add(LSTM(lstm_out, dropout=0.2 ))
#     lstm.add(Dense(3, activation='softmax'))

#     lstm.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
#     print(lstm.summary())

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#     batch_size = 25

#     history = lstm.fit(X_train,
#                         y_train,
#                         epochs = 25,
#                         batch_size=batch_size,
#                         verbose = 2,
#                         validation_data=(X_test, y_test))

#     HIST = pd.DataFrame(history.history)

#     if func == stmr:
#         HIST.to_csv('Model_history/lstm_stmr.csv')
#         lstm.save('Model_chkpts/lstm_stmr.h5')

#     else:
#         HIST.to_csv('Model_history/lstm_lmtzr.csv')
#         lstm.save('Model_chkpts/lstm_lmtzr.h5')

