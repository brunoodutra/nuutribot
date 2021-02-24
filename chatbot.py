#%% imports and librarys 
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
import pickle

import random

from datetime import datetime
import pytz

import logging

from telegram import Update
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
#%% configuração de data e hora do brasil 
T_br = pytz.timezone('America/Sao_Paulo') 
#%% Arquivo base da dieta 
excel_file = 'dieta.xlsx'
dieta = pd.read_excel(excel_file,index_col=0)

#%% carrega a rede LSTM treinada para as perguntas da base de dados 
model_lstm= load_model('model_lstm_nutri.h5')

#%% configuração do tokenizer responsável do pré-processamento das palavras antes de ir pra rede 
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
#%% funções de pré-processamento
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

#%% função de funcionamento do bot com a rede LSTM
def respondebot(text,nome,time):
    global dieta
    resp=read_text(clean_text(text))
    predictions=model_lstm.predict(resp)
    
    x_est=np.zeros([len(predictions[0])])
    x_est2=np.zeros([len(predictions)])

    for i in range(len(predictions)):
        x_est[np.argmax(predictions[0])]=1;
        x_est2[i]=np.argmax([predictions[i]])+1
     
    print(x_est)
    
    if x_est2==1:
        
        if time >= 3 and  time< 12:
            tratamento_hora=' Bom dia'
        elif time >= 12 and  time< 18:
            tratamento_hora=' Boa tarde'
        elif time >=18:
            tratamento_hora=' Boa noite'
        elif time >=0 and  time<3:
            tratamento_hora=' Boa noite'
                
        resposta=("oi "+ nome +tratamento_hora+", o que você deseja?")  
    elif x_est2==2:
        resposta=("Se você quer um plano alimentar novo, pode consultar um profissional humano :D, vou comunicar um amigo humano nutricionista para entrar em contato com você")
    elif x_est2==3:
        resposta=("Ops sou apenas um robô não compreendo esta palavra ainda")
    elif x_est2==4:
        resposta=("Logo mais terei informações para mostrar graficamente a sua evolução") 
    elif x_est2==5:
        dias=dieta.columns.tolist()
        if time >= 3 and  time< 9.5:
            cardapio=dieta[random.choice(dias)][0]
        elif time >= 10 and  time< 12:
            cardapio=dieta[random.choice(dias)][1]
        elif time >= 12 and  time< 14:
            cardapio=dieta[random.choice(dias)][2]
        elif time >=15 and  time <18:
            cardapio=dieta[random.choice(dias)][3]
        elif time >=18 or (time >=0 and  time<3):
            cardapio=dieta[random.choice(dias)][4]
        resposta=cardapio
    elif x_est2==6:
         resposta=("Para você conferir as opções da sua dieta digite por exemplo: \n /help café da manhã \n /help almoço \n /help lanche da manhã \n /help lanche da tarde \n /help jantar") 
    return resposta


#%% configuração de integração com o telegran 
# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)


#%% funções para leitura de frases e palavras do telegran 

# Define a few command handlers. These usually take the two arguments update and
# context. Error handlers also receive the raised TelegramError object in error.
def start(update: Update, context: CallbackContext) -> None:
    """Send a message when the command /start is issued."""
    update.message.reply_text("oi eu sou o nutribot estou aqui para te ajudar."+
                              " Você pode solicitar uma dieta, "+
                              "perguntar pela sua evolução e se informar"+
                              " quais alimentos pode comer agora")


def help_command(update: Update, context: CallbackContext) -> None:
    global dieta, msg
   
    dias=dieta.columns.tolist()
    msg=update.message.text
    msg= msg[6:].lower()
    cardapio=dieta[random.choice(dias)][msg]
    update.message.reply_text(cardapio)

def echo(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    update.message.reply_text(update.message.text)
    
def menssagem(update: Update, context: CallbackContext) -> None:
    """Echo the user message."""
    global T_br, datetime, user
    datetime= datetime.now(T_br)
    user = update.message.from_user
    nome=user.first_name
    
    # print(user.first_name)
    msg=update.message.text
    msg=msg.lower()
    
    if msg=='ok' or msg=='tudo bem' or msg=='certo' or msg=='ta bom' or msg=='blz' or msg=='show' or msg=='oks' or msg=='ta':
        pass
    else:
        print(msg)
        resposta=respondebot(msg,nome,datetime.hour)
        print(resposta)
        update.message.reply_text(resposta)  

def image_handler(update, context):
    group_id = update.message.media_group_id

    if group_id is None:
        file = context.bot.getFile(update.message.photo[0].file_id)
        print ("file_id: " + str(update.message.photo[0].file_id))
        file.download('saved_image.jpg')
def document_handler(update, context):
    print("Documento recebido")
    
    group_id = update.message.media_group_id
    
    if group_id is None:
        file = context.bot.getFile(update.message.document.file_id)
        print ("file_id: " + str(update.message.document.file_id))
        file.download('saved_dieta.xlsx')
#%% Função principal responsável por chamar as funções intermediárias de funcionamento do chatboot
def main():
    """Start the bot."""
    # Create the Updater and pass it your bot's token.
    updater = Updater("1618556516:AAELvOhnilP7Hg8ia_NMzIdB0jfPIyWKL8Y")
    
    # Get the dispatcher to register handlers
    dispatcher = updater.dispatcher

    # on different commands - answer in Telegram
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    updater.dispatcher.add_handler(MessageHandler(Filters.photo, image_handler))
    updater.dispatcher.add_handler(MessageHandler(Filters.document, document_handler))

    
    # handler responsável por verificar menssagens aplicar filtros de texto e encaminhar para a função "menssagem"
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, menssagem))


    # Start the Bot
    updater.start_polling()

    # Run the bot until you press Ctrl-C or the process receives SIGINT,
    # SIGTERM or SIGABRT. This should be used most of the time, since
    # start_polling() is non-blocking and will stop the bot gracefully.
    updater.idle()
    

if __name__ == '__main__':
    main()