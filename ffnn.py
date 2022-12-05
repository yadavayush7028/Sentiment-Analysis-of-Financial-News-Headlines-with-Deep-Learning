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
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Flatten
from keras.utils.np_utils import to_categorical
from keras.utils import to_categorical


def FFNN_model(vocab_size, X, y, embed_dim, num_epochs, batch_size):

    ffnn = Sequential()

    ffnn.add(Embedding(vocab_size, embed_dim, input_length = X.shape[1]))
    ffnn.add(Flatten())
    ffnn.add(Dense(128, input_shape = (embed_dim,), activation='relu'))
    ffnn.add(Dense(64, activation='relu'))
    ffnn.add(Dense(32, activation='relu'))
    ffnn.add(Dense(3, activation='softmax'))

    ffnn.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=[f1])

    y = to_categorical(y, 3)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

    history = ffnn.fit(X_train,
                        y_train,
                        epochs = num_epochs,
                        batch_size=batch_size,
                        verbose = 2,
                        validation_data=(X_test, y_test))

    HIST = pd.DataFrame(history.history)

    preds = ffnn.predict(X_test)

    y_true = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(preds, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    
    return ffnn, HIST, acc, f1_macro

# for func in [stmr, lmtzr]:

#     df = pd.read_csv('combined_data.csv')
#     labels = {'positive':2, 'neutral':1, 'negative':0}
#     df['lbl'] = df['sntmt'].map(lambda x:labels[x])
#     data = process_df(df, func, False)
#     data['tokens'] = data['processed_text'].map(lambda txt: tknzr(txt, False).split(" "))
#     tokens = pd.Series(data['tokens']).values

#     def word_vector(tokens, size):
#         vec = np.zeros(size).reshape((1, size))
#         count = 0
#         for word in tokens:
#             try:
#                 vec += model_w2v.wv[word].reshape((1, size))
#                 count += 1.
#             except KeyError:  # handling the case where the token is not in vocabulary
#                 continue
#         if count != 0:
#             vec /= count
#         return vec
            
#     model_w2v = Word2Vec(
#             tokens,
#             vector_size=300, # desired no. of features/independent variables
#             window=5, # context window size
#             min_count=1, # Ignores all words with total frequency lower than 2.                                  
#             sg = 0, # 1 for skip-gram model
#             hs = 0,
#             negative = 10, # for negative sampling
#         )

#     wordvec_arrays = np.zeros((len(tokens), 300)) 
#     for i in range(len(tokens)):
#         wordvec_arrays[i,:] = word_vector(tokens[i], 300)

#     wordvec_df = pd.DataFrame(wordvec_arrays)

#     X = wordvec_df.values
#     y = data['lbl'].values
#     y = to_categorical(y, 3)

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)

#     batch_size = 50

#     ffnn = Sequential()

#     #ffnn.add(Dense(500, input_shape = (1000,), activation='relu'))
#     ffnn.add(Dense(200, input_shape = (300,), activation='relu'))
#     ffnn.add(Dense(100, activation='relu'))
#     ffnn.add(Dense(50, activation='relu'))
#     ffnn.add(Dense(3, activation='softmax'))

#     ffnn.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     print(ffnn.summary())

#     history = ffnn.fit(X_train,
#                         y_train,
#                         epochs = 25,
#                         batch_size=batch_size,
#                         verbose = 2,
#                         validation_data=(X_test, y_test))

#     HIST = pd.DataFrame(history.history)

#     if func == stmr:
#         HIST.to_csv('Model_history/ffnn_stmr.csv')
#         ffnn.save('Model_chkpts/ffnn_stmr.h5')
#     else:
#         HIST.to_csv('Model_history/ffnn_lmtzr.csv')
#         ffnn.save('Model_chkpts/ffnn_lmtzr.h5')
