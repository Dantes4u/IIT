#!/usr/bin/env python
# coding: utf-8

# In[45]:


#get_ipython().run_line_magic('load_ext', 'autoreload')
#get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path = sys.path[::-1]


# # Импорт

# In[1]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from PIL import Image
from skimage.draw import circle
from skimage import transform, filters, exposure, feature
from sklearn.model_selection import train_test_split
import io
from skimage.color import rgb2gray
from os import listdir
#Создание модели
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import LearningRateScheduler
import json
from glob import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score


# # Константы

# In[2]:
with open("configs.json", "r") as file:
    CONFIGS = json.load(file)

PATH = CONFIGS['PATH']
DESCR = CONFIGS['DESCR']
COLUMNS = ["Номер", "Kiestra_ID", "Kiestra_BARCODE", "Kiestra_SCAN_NR", "Kiestra_CS_ID", "Kiestra_CS_DESCRIPTION", "SCAN_PATH", "Результат"]
GROUPS = ["Unknown", "NoGrowth"]

SIZE = CONFIGS['SIZE']
# Число участников-пациентов и процент тестовой выборки среди них. Пустой профиль означает все профили

# In[3]:



# # Cписок пар изображений для каждого эксперимента

#Пайплайн

chr_nogrowth = []
chr_nogrowth_y = []
blood_nogrowth = []
blood_nogrowth_y = []
chr_unknown = []
chr_unknown_y = []
blood_unknown = []
blood_unknown_y = []

files = glob('/dgx_home/wizard/Naletov/newstep/img_arr/*')
for i in tqdm(files):
    splited = i.split("_")
    #if splited[-2] == 'nogrowth' and splited[-3] == 'chr':
    #    chr_nogrowth.append(np.load(i))
    if splited[-2] == 'nogrowth' and splited[-3] == 'blood':
        blood_nogrowth.append(list(np.load(i)))
    #if splited[-2] == 'unknown' and splited[-3] == 'chr':
    #    chr_unknown.append(np.load(i))
    if splited[-2] == 'unknown' and splited[-3] == 'blood':
        blood_unknown.append(list(np.load(i)))
chr_nogrowth_y = [1]*len(chr_nogrowth)
blood_nogrowth_y = [1]*len(blood_nogrowth)
chr_unknown_y = [0]*len(chr_unknown)
blood_unknown_y = [0]*len(blood_unknown)

X_blood = np.concatenate([blood_nogrowth,blood_unknown])
y_blood = blood_nogrowth_y + blood_unknown_y

X_blood = np.array(X_blood)
y_blood = np.array(y_blood)

X_train_blood, X_test_blood, y_train_blood, y_test_blood = train_test_split(X_blood, y_blood, stratify = y_blood, test_size = 0.2, random_state=42)

print(X_train_blood.shape)
print(len(y_train_blood))
print(X_test_blood.shape)
print(len(y_test_blood))

#train_mean = X_train.mean()
#train_std = X_train.std()
#X_train = (X_train - train_mean)/train_std

#val_mean = X_val_true.mean()
#val_std = X_val_true.std()
#X_val_true = (X_val_true - val_mean)/val_std
#from tensorflow.keras import regularizers

#from keras import layers
#from keras.callbacks import LearningRateScheduler

#learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=2, min_lr=0.00001)



#datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
#    featurewise_center=True,
#   featurewise_std_normalization=True,
#    rotation_range=35,
#    width_shift_range=0.1,
#    height_shift_range=0.1,
#    horizontal_flip=True,
#    rescale=1./255,
#    zoom_range=0.1)
#datagen_valid = tf.keras.preprocessing.image.ImageDataGenerator()

#FREQ = [len(all_subs_un)/all_subs_len,len(all_subs_ng)/all_subs_len]
def mywloss(y_true,y_pred):

    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.math.log(yc)/FREQ[1] + (1- y_true)*tf.math.log(1-yc)/FREQ[0], axis=0)))
    return loss

all_blood_len = len(y_test_blood) + len(y_train_blood)

class_weight = {0:len(blood_nogrowth_y)/len(blood_unknown_y),
                1:1}
print(class_weight)
print(len(blood_nogrowth_y))
print(len(blood_unknown_y))
#datagen_train.fit(X_train)

#train = datagen_train.flow(np.array(X_train), y_train, batch_size=32)
#validation = datagen_valid.flow(np.array(X_val), y_val, batch_size=32)

learning_rate_reduction = LearningRateScheduler(lambda x: 1e-4 * 0.9 ** x)
base_model = tf.keras.applications.Xception(
    weights='imagenet', 
    input_shape=(SIZE, SIZE, 3),
    include_top=False)

inputs = tf.keras.Input(shape=(SIZE, SIZE, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
x = layers.Dropout(0.5)(x)
#x = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
#x = layers.Dropout(0.7)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_pretrained = tf.keras.Model(inputs, outputs)

model_pretrained.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall()])
print(model_pretrained.summary())




model = tf.keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=(5,5), activation='relu', padding = 'same',kernel_initializer='he_uniform', input_shape=(500,500,3)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(128, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#model.add(layers.AveragePooling2D(pool_size=(2,2)))
model.add(layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2,2)))
#model.add(layers.AveragePooling2D(pool_size=(2,2)))
#model.add(layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
#model.add(layers.BatchNormalization())
#model.add(layers.Conv2D(512, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
#model.add(layers.BatchNormalization())
#model.add(layers.MaxPooling2D(pool_size=(2,2)))
#model.add(layers.AveragePooling2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.5))
#model.add(layers.Conv2D(256, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
#model.add(layers.BatchNormalization())
#model.add(layers.AveragePooling2D(pool_size=(2,2)))
#model.add(layers.Dropout(0.5))
#model.add(layers.GlobalAveragePooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))


#model.compile(loss = 'binary_crossentropy',
#              optimizer='adam',
#             metrics=['accuracy'])
#print(model.summary())
model_pretrained.fit(
        X_train_blood, y_train_blood,
        epochs=CONFIGS['epochs'],
        batch_size = 32,
        verbose = 2,
        validation_data = (X_test_blood, y_test_blood),
        class_weight=class_weight,
        callbacks = [learning_rate_reduction])
#model.fit(
#        X_train, y_train,
#        epochs=30,
#        batch_size = 16,
#        verbose = 2,
#        validation_data = (X_val,y_val),
#        callbacks = [learning_rate_reduction])

#y_pred = np.round(model.predict(X_val))
y_pred = np.round(model_pretrained.predict(X_test_blood))
X_err = []
y_res_pred = []
for y_v, y_p, x in zip(y_test_blood, y_pred, X_test_blood):
    if(y_v != y_p):
        X_err.append(x)
        y_res_pred.append(y_p)
np.save(CONFIGS['X_err'], X_err)
np.save(CONFIGS['y_pred'], y_res_pred)
np.save(CONFIGS['original_img'], original_img)
print(accuracy_score(y_pred, y_val))
print(precision_score(y_pred, y_val))
print(recall_score(y_pred, y_val))
#print(accuracy_score(np.round(model.predict(X_val)), y_val))
#model.fit(
#        train,
#        steps_per_epoch = len(X_train) / 32,
#        epochs=30,
#        verbose = 1,
#        validation_data = validation,
#        validation_steps = len(X_val) / 32,
#        callbacks = [learning_rate_reduction])

