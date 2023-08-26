import numpy as np
import pandas as pd
import re
import os

from data_clean import process_df, tknzr, lmtzr, stmr
from utils import f1, vocab_size, max_len, tokenizer
from ffnn import FFNN_model
from rnn import RNN_model
from lstm import LSTM_model
from trfmr import TRFMR_model

df = pd.read_csv('combined_data.csv')
labels = {'positive':2, 'neutral':1, 'negative':0}
df['lbl'] = df['sntmt'].map(lambda x:labels[x])

if not os.path.exists('Model_chkpts'):
    os.makedirs('Model_chkpts')
if not os.path.exists('Model_history'):
    os.makedirs('Model_history')

RESULTS = {'Stop Words': [],
            'Pre-processing': [],
            'Model': [],
            'Macro-F1': [],
            'Accuracy': []}


for rem_sw in [False, True]:

    for pprcs in [tknzr, lmtzr, stmr]:

        data = process_df(df=df.copy(), func=pprcs, rem_sw=rem_sw)

        max_feats = vocab_size(data)
        maxlen = max_len(data)

        X, y = tokenizer(data)

        # # Feed Forward Neural Network

        ffnn_mdl, ffnn_hist, ffnn_acc, ffnn_f1 = FFNN_model(vocab_size=max_feats,
                                                            X = X,
                                                            y = y,
                                                            embed_dim = 256,
                                                            num_epochs = 20,
                                                            batch_size = 10)

        if rem_sw == False:
            RESULTS['Stop Words'].append('Not Removed')
        else:
            RESULTS['Stop Words'].append('Removed')
        
        if pprcs == tknzr:
            RESULTS['Pre-processing'].append('None')
        elif pprcs == lmtzr:
            RESULTS['Pre-processing'].append('Lemmatizing')
        else:
            RESULTS['Pre-processing'].append('Stemming')

        RESULTS['Model'].append('FeedFwdNN')
        RESULTS['Macro-F1'].append(ffnn_f1)
        RESULTS['Accuracy'].append(ffnn_acc)

        ffnn_hist.to_csv(f'Model_history/ffnn_{pprcs.__name__}_{str(int(rem_sw))}.csv', index=False)
        ffnn_mdl.save(f'Model_chkpts/ffnn_{pprcs.__name__}_{str(int(rem_sw))}.h5')

        # # Recurrent Neural Network

        rnn_mdl, rnn_hist, rnn_acc, rnn_f1 = RNN_model(vocab_size = max_feats,
                                                        X = X,
                                                        y = y,
                                                        embed_dim=256,
                                                        out_dim=128,
                                                        num_epochs=20,
                                                        batch_size=10)
        
        if rem_sw == False:
            RESULTS['Stop Words'].append('Not Removed')
        else:
            RESULTS['Stop Words'].append('Removed')
        
        if pprcs == tknzr:
            RESULTS['Pre-processing'].append('None')
        elif pprcs == lmtzr:
            RESULTS['Pre-processing'].append('Lemmatizing')
        else:
            RESULTS['Pre-processing'].append('Stemming')

        RESULTS['Model'].append('RNN')
        RESULTS['Macro-F1'].append(rnn_f1)
        RESULTS['Accuracy'].append(rnn_acc)

        rnn_hist.to_csv(f'Model_history/rnn_{pprcs.__name__}_{str(int(rem_sw))}.csv', index=False)
        rnn_mdl.save(f'Model_chkpts/rnn_{pprcs.__name__}_{str(int(rem_sw))}.h5')

        # # Long Short-Term Memory

        lstm_mdl, lstm_hist, lstm_acc, lstm_f1 = LSTM_model(vocab_size = max_feats,
                                                            X = X,
                                                            y = y,
                                                            embed_dim = 256,
                                                            out_dim = 128,
                                                            num_epochs = 20,
                                                            batch_size = 10)

        if rem_sw == False:
            RESULTS['Stop Words'].append('Not Removed')
        else:
            RESULTS['Stop Words'].append('Removed')
        
        if pprcs == tknzr:
            RESULTS['Pre-processing'].append('None')
        elif pprcs == lmtzr:
            RESULTS['Pre-processing'].append('Lemmatizing')
        else:
            RESULTS['Pre-processing'].append('Stemming')

        RESULTS['Model'].append('LSTM')
        RESULTS['Macro-F1'].append(lstm_f1)
        RESULTS['Accuracy'].append(lstm_acc)

        lstm_hist.to_csv(f'Model_history/lstm_{pprcs.__name__}_{str(int(rem_sw))}.csv', index=False)
        lstm_mdl.save(f'Model_chkpts/lstm_{pprcs.__name__}_{str(int(rem_sw))}.h5')

        # # Transformer

        trfmr_mdl, trfmr_hist, trfmr_acc, trfmr_f1 = TRFMR_model(vocab_size = max_feats,
                                                                X = X,
                                                                y = y,
                                                                maxlen = maxlen,
                                                                embed_dim = 256,
                                                                num_heads = 3,
                                                                out_dim = 128,
                                                                num_epochs = 20,
                                                                batch_size = 10)      
        
        if rem_sw == False:
            RESULTS['Stop Words'].append('Not Removed')
        else:
            RESULTS['Stop Words'].append('Removed')
        
        if pprcs == tknzr:
            RESULTS['Pre-processing'].append('None')
        elif pprcs == lmtzr:
            RESULTS['Pre-processing'].append('Lemmatizing')
        else:
            RESULTS['Pre-processing'].append('Stemming')

        RESULTS['Model'].append('Transformer')
        RESULTS['Macro-F1'].append(trfmr_f1)
        RESULTS['Accuracy'].append(trfmr_acc)

        trfmr_hist.to_csv(f'Model_history/trfmr_{pprcs.__name__}_{str(int(rem_sw))}.csv', index=False)
        trfmr_mdl.save(f'Model_chkpts/trfmr_{pprcs.__name__}_{str(int(rem_sw))}.h5')

        results_phase2 = pd.DataFrame(RESULTS)
        results_phase2.to_csv('RESULTS1.csv', index=False)

