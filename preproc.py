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
from sklearn.utils import shuffle
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
import joblib


# # Константы

# In[2]:
with open("configs.json", "r") as file:
    CONFIGS = json.load(file)

SIZE = CONFIGS['SIZE']

from skimage import io
from skimage.transform import resize
"""
rain = './rainy-image-dataset/rainy image/*'
rain = glob(rain)
gr_truth = './rainy-image-dataset/ground truth/*'
gr_truth = glob(gr_truth)
X = []
y = []
for i in tqdm(rain[10000:]): 
    for j in gr_truth:
        if i.split('/')[-1][:-4].split('_')[0] == j.split('/')[-1][:-4]:
            img = io.imread(j)
            img = resize(img,(SIZE,SIZE,3))
            y.append(img.tolist())
    img = io.imread(i)
    img = resize(img,(SIZE,SIZE,3))
    X.append(img.tolist())
X = np.array(X)
y = np.array(y)

np.save(CONFIGS['X_prep_3'], X)
np.save(CONFIGS['y_prep_3'], y)
"""
def loss(y_true,y_pred):
    norm = tf.norm(y_true - y_pred, ord="fro", axis=[-2, -1])
    loss = tf.reduce_mean(norm, 1)
    return loss

X = np.load(f'./{CONFIGS["X_prep"]}.npy')
y = np.load(f'./{CONFIGS["y_prep"]}.npy')

learning_rate_reduction = LearningRateScheduler(lambda x: 1e-2 * 0.9 ** x)
model = tf.keras.Sequential()
model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding = 'same',kernel_initializer='he_uniform', input_shape=(SIZE,SIZE,3)))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(64, kernel_size=(5,5), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(16, kernel_size=(3,3), activation='relu', padding = 'same', kernel_initializer='he_uniform'))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(3, kernel_size=(3,3), activation='relu', padding = 'same',kernel_initializer='he_uniform'))


model.compile(loss = loss,
              optimizer='adam',
             metrics=['mse'])
model.fit(
        X, y,
        epochs=20,
        batch_size = 16,
        verbose = 2,
        callbacks = [learning_rate_reduction])
del X
del y

X = np.load(f'./{CONFIGS["X_prep_2"]}.npy')
y = np.load(f'./{CONFIGS["y_prep_2"]}.npy')

model.fit(
        X, y,
        epochs=20,
        batch_size = 16,
        verbose = 2)
        
del X
del y

X = np.load(f'./{CONFIGS["X_prep_3"]}.npy')
y = np.load(f'./{CONFIGS["y_prep_3"]}.npy')

model.fit(
        X, y,
        epochs=20,
        batch_size = 16,
        verbose = 2)
        
del X
del y

model.save(CONFIGS['prep_model'])