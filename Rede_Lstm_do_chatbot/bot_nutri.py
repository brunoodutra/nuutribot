# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 12:04:44 2021

@author: Bruno Dutra
"""


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('dark_background')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from keras.models import Model, load_model
import pickle
from datetime import datetime
import pytz

tz_NY = pytz.timezone('America/Sao_Paulo') 



model_lstm= load_model('model_lstm_nutri.h5')


with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

replace_list = {r"á": 'a',
                r"ê": 'e',
                r"ç": 'c',
                r"â":  'a',
                r"'ve": ' have',
                r"can't": 'can not',
                r"cannot": 'can not',
                r"shan’t": 'shall not',
                r"n't": ' not',
                r"'d": ' would',
                r"'ll": ' will',
                r"'scuse": 'excuse',
                ',': ' ',
                '.': ' ',
                '!': ' ',
                '?': ' ',
                '\s+': ' '}
def clean_text(text):
    text = text.lower()
    for s in replace_list:
        text = text.replace(s, replace_list[s])
    text = ' '.join(text.split())
    return text

def read_text(text_):
    X_text=pd.Series(text_) 
    text = tokenizer.texts_to_sequences(X_text)
    text = pad_sequences(text, maxlen = 6)  
    return text

text=''

print("Sou o Nutri-bot, estou pronto para te ajudar")

while text != 'exit':

    text=input()
    resp=read_text(clean_text(text))
    predictions=model_lstm.predict(resp)
    
    x_est=np.zeros([len(predictions[0])])
    x_est2=np.zeros([len(predictions)])

    for i in range(len(predictions)):
        x_est[np.argmax(predictions[0])]=1;
        x_est2[i]=np.argmax([predictions[i]])+1
     
    print(x_est)
    print(predictions)
    
    if x_est2==1:
        
        datetime= datetime.now(tz_NY)
        
        if datetime.hour >= 3 and  datetime.hour< 12:
            tratamento_hora='Bom dia'
        elif datetime.hour >= 12 and  datetime.hour< 18:
            tratamento_hora='Boa tarde'
        elif datetime.hour >=18:
            tratamento_hora='Boa noite'
        elif datetime.hour >=0 and  datetime.hour<3:
            tratamento_hora='Boa noite'
                    
        print("oi, "+tratamento_hora+", o que vocês deseja?")
        
    elif x_est2==2:
        print("Ok uma dieta saindo em instantes")
    elif x_est2==3:
        print("Ops palavra feia, me respeite por favor ")
    elif x_est2==4:
        print("Evolução de 100% papai") 
        
        



