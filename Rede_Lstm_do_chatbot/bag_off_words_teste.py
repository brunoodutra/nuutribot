# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 02:07:00 2021

@author: Bruno Dutra
"""
# https://towardsdatascience.com/machine-learning-recurrent-neural-networks-and-long-short-term-memory-lstm-python-keras-example-86001ceaaebc

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
import pickle


opc_=['o que devo almoçar','o que posso lanchar','o que devo lanchar',
      'o que devo comer de almoço','o que devo comer de janta','o que devo comer de lanche',
      'outra opção','o que posso comer de lanche','lanche','o que posso comer no jantar',
      'o que posso comer no almoço','almoço','almoçar', 
      'café da manhã','opções de comida','consultar refeições',
      'opções de jantar','opções de almoço','opções de lanche']

comida_agora=['outra opção','comer','comidinha','lanchinho','o que posso comer agora',
              'quais alimentos devo consumir','qual a minha dieta de agora',
              'o que devo','o que posso','o que comer','opções para comer',
              'opções de alimentos','opções de dieta','o que devo comer agora']

evolution=['quero saber a minha evolução','quero o meu resultado',
           'quero os meus resultados','resultados','resultado','rendimento',
           'evolução','dando certo','quero saber minha evolução','meu rendimento',
           'quero saber se ta dando certo']

palavroes=['fdp','puta','viadinho','caralho','porra','merda','gayzão','gay','imprestável','viadão']

intro_dieta=['plano alimentar','emagrecer','perder peso','quero uma dieta ', 
             'dieta',' gostaria de emagrecer', 'passe uma dieta', 'quais dietas tens',
             'preciso de uma perder peso', 'me ajude a fazer uma dieta']

intro=['Boa tarde','bom dia','boa noite','fale','oi','oie tudo bem','tudo bem',
       'ei','como você está','quero bolo','olá','como voce esta','e aew','fala'] 

train=  intro+ intro_dieta + palavroes+ evolution + comida_agora + opc_;

T_6=np.ones(len(opc_))*5
T_5=np.ones(len(comida_agora))*4
T_4=np.ones(len(evolution))*3
T_3=np.ones(len(palavroes))*2
T_2=np.ones(len(intro_dieta))*1
T_1=np.ones(len(intro))*0

target  = T_1.tolist()+T_2.tolist()+T_3.tolist()+T_4.tolist()+T_5.tolist()+T_6.tolist();

# train = ['dieta','ajuda','emagrecer','perder','plano alimentar','plano','alimentar'
#          ,'tudo bem','ei','e ai','olá','oi','hey','hello']

# target  = [1,1,1,1,1,1,1,0,0,0,0,0,0,0]

samples={'train':train,
         'target':target}
df_samples = pd.DataFrame(samples, columns = ['train', 'target'])

#print (df)

#%%###########################################################################

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

X_train_ = df_samples['train'].apply(lambda p: clean_text(p))
#X_train_=df_samples['train'];
phrase_len = X_train_.apply(lambda p: len(p.split(' ')))
max_phrase_len = phrase_len.max()
# max_phrase_len = 100;

print('max phrase len: {0}'.format(max_phrase_len))
plt.figure(figsize = (10, 8))
plt.hist(phrase_len, alpha = 0.2, density = True)
plt.xlabel('phrase len')
plt.ylabel('probability')
plt.grid(alpha = 0.25)

y_train = df_samples['target']
max_words = 100

tokenizer = Tokenizer(
    num_words = max_words,
    filters = '"#$%&()*+-/:;<=>@[\]^_`{|}~'
)

tokenizer.fit_on_texts(X_train_)

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

X_train = tokenizer.texts_to_sequences(X_train_)

one_hot_results = tokenizer.texts_to_matrix(X_train_, mode='binary')
word_index = tokenizer.word_index

X_train = pad_sequences(X_train, maxlen = max_phrase_len)
y_train = to_categorical(y_train)
#%%###########################################################################
batch_size = 2**8
epochs = 100

number_outuputs=np.shape(y_train)[1];

# model_lstm = Sequential()
# model_lstm.add(Embedding(max_words, batch_size))
# model_lstm.add(LSTM(256))
# model_lstm.add(Dense(number_outuputs, activation='sigmoid'))


model_lstm = Sequential()
model_lstm.add(Embedding(input_dim = max_words, output_dim = 2**9, input_length = max_phrase_len))
model_lstm.add(SpatialDropout1D(0.3))
model_lstm.add(LSTM(2**8, dropout = 0.3, recurrent_dropout = 0.3))
model_lstm.add(Dense(2**7, activation = 'relu'))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(number_outuputs, activation = 'softmax'))

model_lstm.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


history = model_lstm.fit(
    X_train,
    y_train,
    validation_split = 0.1,
    epochs = epochs,
    batch_size = batch_size
)

plt.clf()
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'yo', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'g', label='Training acc')
plt.plot(epochs, val_acc, 'yo', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

model_lstm.predict(X_train)

model_lstm.save('model_lstm_nutri.h5')
model_lstm.save_weights('pre_trained_glove_model.h5')
#%%###########################################################################
predictions=model_lstm.predict(X_train)
x_est=np.zeros([len(predictions)])
for i in range(len(predictions)):
    x_est[i]=np.argmax([predictions[i]])

def read_text(text_):
    X_text=pd.Series(text_) 
    text = tokenizer.texts_to_sequences(X_text)
    text = pad_sequences(text, maxlen = 6)  
    return text

text=''

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
    if x_est2==1:
        print("oi, Bom dia, o que vocês deseja?")
    elif x_est2==2:
        print("Ok uma dieta saindo em instantes")
    elif x_est2==3:
        print("Ops palavra feia, me respeite por favor ")
    elif x_est2==4:
        print("Evolução de 100% papai") 
        


