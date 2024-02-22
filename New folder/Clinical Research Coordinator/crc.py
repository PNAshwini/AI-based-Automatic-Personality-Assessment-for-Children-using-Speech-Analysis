import cv2 
import os
import sys
import gc
import librosa
from scipy.stats import zscore

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
import hashlib
import socket
from flask_mysqldb import MySQL
from flask import *
import numpy as np
import os
from functools import wraps
import webbrowser
import ctypes
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
from flask_mysqldb import MySQL
from tqdm import tqdm
import hashlib
import controller as ct
import cv2
import mysql.connector as mssql
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
dic  =  {1:'Neutral', 2:'Extraversion', 3:'Conscientiousness', 4:'Neuroticism', 5:'Neuroticism and Agreeableness', 6:'Fearfulness', 7:'Obsessive/Compulsive', 8:'Openness'}
def mel_spectrogram(y, sr=16000, n_fft=512, win_length=256, hop_length=128, window='hamming', n_mels=128, fmax=4000):
    
    # Compute spectogram
    mel_spect = np.abs(librosa.stft(y, n_fft=n_fft, window=window, win_length=win_length, hop_length=hop_length)) ** 2
    
    # Compute mel spectrogram
    mel_spect = librosa.feature.melspectrogram(S=mel_spect, sr=sr, n_mels=n_mels, fmax=fmax)
    
    # Compute log-mel spectrogram
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    
    return mel_spect
def predict(path):
    model = load_model("../Model/Emotion_Model.h5")
    s = []
    # Sample rate (16.0 kHz)
    sample_rate = 16000     
    # Max pad lenght (3.0 sec)
    max_pad_len = 49100
    X, sample_rate = librosa.load(path,duration=3,offset=0.5)
    sample_rate = np.array(sample_rate)
    y = zscore(X)  
    # Padding or truncated signal 
    if len(y) < max_pad_len:    
        y_padded = np.zeros(max_pad_len)
        y_padded[:len(y)] = y
        y = y_padded
    elif len(y) > max_pad_len:
        y = np.asarray(y[:max_pad_len])

    # Add to signal list
    s.append(y)
    mel_spect = np.asarray(list(map(mel_spectrogram, s)))
    win_ts = 128
    hop_ts = 64

# Split spectrogram into frames
    def frame(x, win_step=128, win_size=64):
        nb_frames = 1 + int((x.shape[2] - win_size) / win_step)
        frames = np.zeros((x.shape[0], nb_frames, x.shape[1], win_size)).astype(np.float32)
        for t in range(nb_frames):
            frames[:,t,:,:] = np.copy(x[:,:,(t * win_step):(t * win_step + win_size)]).astype(np.float32)
        return frames

    # Frame for TimeDistributed model
    x = frame(mel_spect, hop_ts, win_ts)
    x = x.reshape(x.shape[0], x.shape[1] , x.shape[2], x.shape[3], 1)
    preds = model.predict(x)
    preds=preds.argmax(axis=1)
    return dic[int(preds)]

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

app=Flask(__name__, template_folder='templates', static_folder='static')
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']='taylor@1989'
app.config['MYSQL_DB']='weed'
app.config['MYSQL_CURSORCLASS']='DictCursor'
mysql=MySQL(app) 
