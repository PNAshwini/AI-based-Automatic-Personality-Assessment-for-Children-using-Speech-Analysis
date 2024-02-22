import cv2 
import os, sys
import random
import string
import os
import gc
import mysql.connector as mssql
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import hashlib
import socket
global model
import librosa
def getMachine_addr():
	os_type = sys.platform.lower()
	command = "wmic bios get serialnumber"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def getUUID_addr():
	os_type = sys.platform.lower()
	command = "wmic path win32_computersystemproduct get uuid"
	return os.popen(command).read().replace("\n","").replace("	","").replace(" ","")

def extract_command_result(key,string):
    substring = key
    index = string.find(substring)
    result = string[index + len(substring):]
    result = result.replace(" ","")
    result = result.replace("-","")
    return result
def train():
    
        # %%


    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

    import matplotlib.pyplot as plt
    import librosa
    import librosa.display
    from IPython.display import Audio, display

    # Input data files are available in the read-only "../input/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

    import os

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import itertools

    import librosa
    import librosa.display

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder,StandardScaler
    from sklearn import preprocessing
    from keras.layers import Dense, BatchNormalization, Dropout, LSTM
    from keras.models import Sequential
    from keras.utils import to_categorical
    from keras import callbacks
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

    from tqdm import tqdm
    from IPython.display import Audio, display

    from keras.utils import np_utils

    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

    # %% [markdown]
    # # Loding The Data

    # %%
    files_Path='../Dataset/'

    c_emotions = {
        'NEU':'neutral',
        'HAP':'happy',
        'SAD':'sad',
        'ANG':'angry',
        'FEA':'fear',
        'DIS':'disgust'}

    c_file = []
    for wav in os.listdir(files_Path):
        emo = wav.partition(".wav")[0].split('_')
        emotion = c_emotions[emo[2]]
        c_file.append((files_Path+'/'+wav,emotion))
        
    data_df = pd.DataFrame(c_file, columns = ['File_path', 'Emotion'])
    data_df.to_csv('data_df.csv')
    data_df.shape
    data_df.head(10)    

    # %% [markdown]
    # # Data Analysis

    # %%
    import matplotlib.pyplot as plt
    import librosa

    def create_waveplot(data, sampling_rate, emotion):
        plt.figure(figsize=(10, 3))
        plt.title(f'Waveplot for audio with {emotion} emotion', size=15)
        plt.plot(data)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.show()

    for emotion in c_emotions.values():
        # get the path of the first happy record
        path = (data_df[data_df.Emotion == emotion].iloc[0])[0]
        data, sampling_rate = librosa.load(path)
        create_waveplot(data,sampling_rate,emotion)
        # create_spectrogram(data, sampling_rate, emotion)
        display(Audio(path))


    # %%
    plt.style.use("ggplot")
    cols= ["#6dbf9f","#774571","#234554","#45d874","#864525","#691285"]
    plt.title("Count of emotions")
    sns.countplot(x = data_df["Emotion"], palette= cols)
    sns.despine(top = True, right = True, left = False, bottom = False)

    # %% [markdown]
    # note: the data seems to be somewhat balanced.
    # 

    # %% [markdown]
    # # Data Processing

    # %%
    def add_noise(data,random=False,rate=0.035,threshold=0.075):
        if random:
            rate=np.random.random()*threshold
        noise=rate*np.random.uniform()*np.amax(data)
        augmented_data=data+noise*np.random.normal(size=data.shape[0])
        return augmented_data

    def pitching(data,sr,pitch_factor=0.7,random=False):
        if random:
            pitch_factor=np.random.random() * pitch_factor
        return librosa.effects.pitch_shift(y=data,sr=sr,n_steps=pitch_factor)
    #-------------------------------------------------------------------------------------------
    def zcr(data,frame_length,hop_length):
        zcr=librosa.feature.zero_crossing_rate(y=data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(zcr)
    def rmse(data,frame_length=2048,hop_length=512):
        rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
        return np.squeeze(rmse)
    def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
        mfcc=librosa.feature.mfcc(y=data,sr=sr)
        return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)
    #-------------------------------------------------------------------------------------------
    def extract_features(data,sr,frame_length=2048,hop_length=512):
        result=np.array([])
        
        result=np.hstack((result,
                        zcr(data,frame_length,hop_length),
                        rmse(data,frame_length,hop_length),
                        mfcc(data,sr,frame_length,hop_length)
                        ))
        return result

    def get_features(path,duration=2.5, offset=0.6):
        data,sr=librosa.load(path,duration=duration,offset=offset)
        aud=extract_features(data,sr)
        audio=np.array(aud)
        
        noised_audio=add_noise(data,random=True)
        aud2=extract_features(noised_audio,sr)
        audio=np.vstack((audio,aud2))
        
        pitched_audio=pitching(data,sr,random=True)
        aud3=extract_features(pitched_audio,sr)
        audio=np.vstack((audio,aud3))
        
        return audio
    #-------------------------------------------------------------------------------------------
    X, Y = [], []

    for path, emotion, index in zip(data_df.File_path, data_df.Emotion, range(data_df.File_path.shape[0])):
        features = get_features(path)
        if index % 500 == 0:
            print(f'{index} audio has been processed')
        for i in features:
            X.append(i)
            Y.append(emotion)

    print('Done')


    # %%
    processed_data_path='./processed_data.csv'

    # %%
    extract=pd.DataFrame(X)
    extract['Emotion']=Y
    extract.to_csv(processed_data_path,index=False)
    extract.head(10)

    # %%
    df=pd.read_csv(processed_data_path)
    df.shape

    df=df.fillna(0)
    print(df.isna().any())
    df.shape

    # %%
    df.tail(10)

    # %%
    X=df.drop(labels='Emotion',axis=1)
    Y=df['Emotion']
    lb=LabelEncoder()
    Y=np_utils.to_categorical(lb.fit_transform(Y))
    print(lb.classes_)
    Y

    # %%
    X_train,X_test,y_train,y_test = train_test_split(X,Y,random_state=32,test_size=0.2,shuffle=True)
    X_train.shape,X_test.shape,y_train.shape,y_test.shape

    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,random_state=32,test_size=0.1,shuffle=True)
    X_train.shape, X_test.shape, X_val.shape, y_train.shape,y_test.shape,y_val.shape

    # %%
    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)
    X_test=scaler.transform(X_test)
    X_val=scaler.transform(X_val)
    X_train.shape,X_test.shape,X_val.shape,y_train.shape,y_test.shape,y_val.shape

    # %%
    X_train=np.expand_dims(X_train,axis=2)
    X_val=np.expand_dims(X_val,axis=2)
    X_test=np.expand_dims(X_test,axis=2)
    X_train.shape, X_test.shape, X_val.shape

    # %% [markdown]
    # # Model Building

    # %%
    import tensorflow.keras.layers as L
    import tensorflow as tf
    from keras.callbacks import ReduceLROnPlateau
    early_stop = callbacks.EarlyStopping(monitor='val_loss', mode='auto', patience=5, restore_best_weights=True)
    lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.1, min_lr=0.00001)
    EPOCH=50
    BATCH_SIZE=30

    model=tf.keras.Sequential([
        L.Conv1D(32,kernel_size=6, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
        L.MaxPool1D(pool_size=5,strides=2,padding='same'),
        L.Conv1D(64,kernel_size=6,strides=1,padding='same',activation='relu'),
        L.MaxPool1D(pool_size=5,strides=2,padding='same'),
        L.Conv1D(128,kernel_size=6,strides=1,padding='same',activation='relu'),
        L.MaxPool1D(pool_size=5,strides=2,padding='same'),
        L.Flatten(),
        L.Dense(256,activation='relu'),
        L.Dropout(0.5),
        L.Dense(6,activation='softmax')
    ])
    model.compile(optimizer='nadam',loss='categorical_crossentropy',metrics='accuracy')
    history=model.fit(X_train, y_train, epochs=EPOCH, validation_data=(X_val,y_val), batch_size=BATCH_SIZE,callbacks=[early_stop,lr_reduction])

    # %%
    val_accuracy = np.mean(history.history['val_accuracy'])
    print("\n%s: %.2f%%" % ('val_accuracy', val_accuracy*100))

    # %%
    history_df = pd.DataFrame(history.history)

    plt.plot(history_df.loc[:, ['loss']], "#6daa9f", label='Training loss')
    plt.plot(history_df.loc[:, ['val_loss']],"#774571", label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc="best")

    plt.show()

    # %%
    history_df = pd.DataFrame(history.history)

    plt.plot(history_df.loc[:, ['accuracy']], "#6daa9f", label='Training accuracy')
    plt.plot(history_df.loc[:, ['val_accuracy']], "#774571", label='Validation accuracy')

    plt.title('Training and Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # %%
    # Predicting the test set results
    y_pred = model.predict(X_test)
    y_pred = (y_pred > 0.5)
    np.set_printoptions()

    # %% [markdown]
    # # Conclusion

    # %%
    import seaborn as sns
    # confusion matrix
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    conf=confusion_matrix(y_test_labels,y_pred_labels)
    cmap1 = sns.diverging_palette(275,100,  s=40, l=65, n=6)
    cm=pd.DataFrame(
        conf,index=[i for i in c_emotions.values()],
        columns=[i for i in c_emotions.values()]
    )
    plt.figure(figsize=(12,7))
    ax=sns.heatmap(cm,annot=True,fmt='d')
    ax.set_title(f'confusion matrix for model ')
    plt.show()




    # %%

global files_Path
files_Path ='../Dataset/english_words_sentences/10_M_nonNative/train'
global data_df
c_emotions = {
    'NEU':'neutral',
    'HAP':'happy',
    'SAD':'sad',
    'ANG':'angry',
    'FEA':'fear',
    'DIS':'disgust'}

def preprocess():
    global data_df
    c_file = []
    for wav in os.listdir(files_Path):
        emo = wav.partition(".wav")[0].split('_')
        emotion = c_emotions[emo[2]]
        c_file.append((files_Path+'/'+wav,emotion))
        
    data_df = pd.DataFrame(c_file, columns = ['File_path', 'Emotion'])
    data_df.to_csv('data_df.csv')


def add_noise(data,random=False,rate=0.035,threshold=0.075):
    if random:
        rate=np.random.random()*threshold
    noise=rate*np.random.uniform()*np.amax(data)
    augmented_data=data+noise*np.random.normal(size=data.shape[0])
    return augmented_data

def pitching(data,sr,pitch_factor=0.7,random=False):
    if random:
        pitch_factor=np.random.random() * pitch_factor
    return librosa.effects.pitch_shift(y=data,sr=sr,n_steps=pitch_factor)
#-------------------------------------------------------------------------------------------
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)
def rmse(data,frame_length=2048,hop_length=512):
    rmse=librosa.feature.rms(y=data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)
def mfcc(data,sr,frame_length=2048,hop_length=512,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y=data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)
#-------------------------------------------------------------------------------------------
def extract_features(data,sr,frame_length=2048,hop_length=512):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr,frame_length,hop_length)
                     ))
    return result

def get_features(path,duration=2.5, offset=0.6):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_features(data,sr)
    audio=np.array(aud)
    
    noised_audio=add_noise(data,random=True)
    aud2=extract_features(noised_audio,sr)
    audio=np.vstack((audio,aud2))
    
    pitched_audio=pitching(data,sr,random=True)
    aud3=extract_features(pitched_audio,sr)
    audio=np.vstack((audio,aud3))
    
    return audio
#-------------------------------------------------------------------------------------------
X, Y = [], []

def extract():    
    for path, emotion, index in zip(data_df.File_path, data_df.Emotion, range(data_df.File_path.shape[0])):
        features = get_features(path)
        if index % 500 == 0:
            print(f'{index} audio has been processed')
        for i in features:
            X.append(i)
            Y.append(emotion)

print('Done')
    
def save_model():
    global model
    model_save= load_model('../Model/Emotion_Model.h5')
    if(model_save):
        return True
    else:
        return False


def plot_accuracy():
    image = cv2.imread('../Plots/loss.png')
    return image

def plot_loss():
    image = cv2.imread('../Plots/loss.png')
    return image

def md5(input_string):
    md5_hash = hashlib.md5()
    md5_hash.update(input_string.encode('utf-8'))
    return md5_hash.hexdigest()

def get_ip_address_of_host():
    mySocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        mySocket.connect(('10.255.255.255', 1))
        myIPLAN = mySocket.getsockname()[0]
    except:
        myIPLAN = '127.0.0.1'
    finally:
        mySocket.close()
    return myIPLAN

def key_validate(str):
    conn = mssql.connect(
        user='root', password='root', host='localhost', database='speech'
        )
    cur = conn.cursor()
    private_key = extract_command_result("SerialNumber",getMachine_addr()) + extract_command_result("UUID",getUUID_addr())
    if private_key in str:
        cur.execute("select * from SOFTKEY where private_key = %s and public_key = %s",(md5(private_key),md5(extract_command_result(private_key,str))))
        data=cur.fetchone()
        if data:
            return True
        else:
            return False
    else:
        return False

