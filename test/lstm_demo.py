import pandas as pd
import numpy as np
import os
from keras.models import Model
from keras.layers import Dense,Dropout,Input,concatenate
from keras.layers.embeddings import Embedding
# from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder
from keras.callbacks import EarlyStopping
from keras.layers.recurrent import LSTM
from keras.utils.np_utils import *
import random

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
# 指定第一块GPU可用
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
KTF.set_session(sess)

def get_list():
    dataset='mr'
    f = open('../data/corpus/' + dataset + '.clean.txt', 'r')
    data = f.readlines()

    f = open('../data/' + dataset + '_shuffle.txt', 'r')
    lines = f.readlines()
    datas=[]
    labels=[]
    test_datas=[]
    test_labels=[]
    for line in lines:
        t=line.split('\t')
        if t[1]=='train':
            datas.append(data[int(t[0])])
            labels.append(int(t[2][0]))
        if t[1]=='test':
            test_datas.append(data[int(t[0])])
            test_labels.append(int(t[2][0]))
    f.close()
    data_lst=[]
    for data in datas:
        t=data.split()
        data_lst.append(' '.join(t))

    test_data_lst=[]
    for data in test_datas:
        t=data.split()
        test_data_lst.append(' '.join(t))

    return data_lst,labels,test_data_lst,test_labels

def points2txt(data,roi_file='logs.txt'):#将鼠标读取的roi坐标点存储到txt文件中
    f = open(roi_file, 'a')
    f.write(data+"\n")
    f.close()

def proceed_dict1():
    data, target,test_data,test_target=get_list()
    data, target = np.array(data), np.array(target)
    test_data, test_target = np.array(test_data), np.array(test_target)
    train_num = 7108-710
    train_withdraw = int(train_num * 0.025)
    idx = random.sample(range(0, train_num), train_withdraw)
    X_val = data[-710:]
    y_val = target[-710:]
    data = data[:-710]
    target = target[:-710]
    X_train = [data[i] for i in idx]
    y_train = [target[i] for i in idx]
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test = test_data
    y_test = test_target

    maxlen = 200
    num_words=18000
    token = Tokenizer(num_words=num_words)
    token.fit_on_texts(data)

    print(len(token.word_index))
    x_train_seq = token.texts_to_sequences(X_train)
    x_val_seq = token.texts_to_sequences(X_val)
    x_test_seq = token.texts_to_sequences(X_test)

    x_train = sequence.pad_sequences(x_train_seq, maxlen=maxlen)
    x_val = sequence.pad_sequences(x_val_seq, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test_seq, maxlen=maxlen)

    callback = EarlyStopping(monitor="val_loss", patience=10, verbose=0, mode='auto')
    # 主要损失函数
    main_input = Input(shape=(maxlen,))
    x1 = Embedding(input_dim=num_words, output_dim=32, input_length=np.shape(x_train)[1], name='word_embedding')(main_input)
    x1 = Dropout(0.5)(x1)
    x1 = LSTM(32,name='LSTM')(x1)
    x1 = Dense(16, activation="relu")(x1)
    x1 = Dropout(0.5)(x1)
    auxiliary_output = Dense(1, activation="sigmoid", name='auxiliary_output')(x1)
    model = Model(inputs=[main_input], outputs=[auxiliary_output])
    print(model.summary())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit([x_train], [y_train], epochs=500, batch_size=500, validation_data=([x_val], [y_val]), verbose=2,
              shuffle=False,callbacks=[callback])
    print('ok')
    from metrics import classify_evalue
    y_pre = model.predict(x_test)
    y_pred = [int(round(i[0])) for i in y_pre]
    print(classify_evalue(y_test, y_pred, 'acc'))
    print(classify_evalue(y_test, y_pred, 'f1'))
    print(classify_evalue(y_test, y_pred, 'roc_auc'))
    test_acc=classify_evalue(y_test, y_pred, 'acc')
    points2txt(str(test_acc))

if __name__=='__main__':
    for _ in range(5):
        proceed_dict1()

