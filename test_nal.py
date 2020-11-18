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

# # Константы

# In[2]:

PATH = "./"
DESCR = "./ScansDescriptionWithPathCS.csv"
COLUMNS = ["Номер", "Kiestra_ID", "Kiestra_BARCODE", "Kiestra_SCAN_NR", "Kiestra_CS_ID", "Kiestra_CS_DESCRIPTION", "SCAN_PATH", "Результат"]
GROUPS = ["Unknown", "NoGrowth"]


# Число участников-пациентов и процент тестовой выборки среди них. Пустой профиль означает все профили

# In[3]:


PROFILE = []#["Blood agar Colonies"]
CAP = 400 #100
TRAIN = 1.0 # Если дробное то процент от CAP


# # Полезные функции

# Добавляем в данные булев столбец-результат

# In[4]:


def add_result(table, name="RESULT"):
    table = table.copy()
    table[name] = [False if "NoGrowth" in x else True for x in table["SCAN_PATH"]]
    return table


# Способ выборки тестовой и тренировочной

# In[5]:


def select(index, cap, train):
    if type(train) == int:
        l = train
    else:
        l = int(cap * train)
    return index[:l], index[l:cap]


# Подсчет ошибок

# In[6]:


def count(patients, model, error_condition=lambda x: x > 0):
    errors = 0
    files = []
    for index in tqdm(patients):
        filenames, vectors = build_vectors([index], silent=True)
        if len(vectors) == 0:
            patients = patients.delete(list(patients).index(index))
            continue
        pred = model.predict(vectors)
        pred = np.rint(pred)
        if error_condition(np.count_nonzero(pred)):
            errors += 1
            files += [filenames[i] for i in range(len(filenames)) if error_condition(pred[i])]
    return patients, errors, files


# Подсчет ошибок в общем

# In[7]:


def eval(patients_0, patients_1, model):
    _, vectors_0 = build_vectors(patients_0, silent=True)
    _, vectors_1 = build_vectors(patients_1, silent=True)
    X = np.concatenate((vectors_0, vectors_1))
    Y = np.concatenate((np.zeros(len(vectors_0)), np.ones(len(vectors_1))))
    result = np.rint(model.predict(X).flatten())
    errors = np.count_nonzero(result - Y)
    return errors, result


# Отображаем картинки

# In[8]:


def show_images(images, cols = 1, titles = None):
    """Display a list of images in a single figure with matplotlib.
    
    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    
    cols (Default = 1): Number of columns in figure (number of rows is 
                        set to np.ceil(n_images/float(cols))).
    
    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images * 2)
    plt.show()

# # Функция предобработки

# Параметры - исходный размер, размер границ и размер до которого ресайзим

# In[41]:


D = 2000
RIM = 300
SIZE = 500


# Функция предобработки, сглаживание и эквализация гистограммы закоментированы, тут только ресайз и обрезка

# In[42]:


CENTER = D/2 - RIM
MASK = np.zeros((D, D, 3))[RIM:-RIM, RIM:-RIM]
MASK[circle(CENTER, CENTER, CENTER)] = 1
def PREPROCESS(image):
    
    # Cutting circle
    image = image[RIM:-RIM, RIM:-RIM]
    image = image * MASK
    
    # Equalizing histogram
    #image /= 255
    #image = exposure.equalize_adapthist(image)
    
    # Filtering
    #image = filters.gaussian(image, sigma=1.0, multichannel=True)
    
    # Resizing
    image = transform.resize(image, (SIZE, SIZE,3))
    
    return image.astype(int)

# # Функция считывания картинок и группировки

# Считываем по каждому участнику списка

# In[13]:


def read(patients, select=lambda x: x, silent=False):
    IMAGES = []
    filenames = []
    groups = []
    wrap = lambda x: x if silent else tqdm(x)
    # Пациенты
    for index in wrap(patients):
        try:
            un_path = ("./Unknown")
            ng_path = ("./NoGrowth")
            if index.replace(" ", "") in listdir(un_path) or index.replace(" ", "") in listdir(ng_path):
                group = PATIENTS.get_group(index)
                # Чашки
                one_person = []
                one_person_images = []
                for barcode, btable in group.groupby("Kiestra_BARCODE"):
                    # Профили
                    for profile, ptable in btable.groupby("Kiestra_CS_DESCRIPTION"):
                        # Пути
                        if("chrom" not in profile.lower()) and ("custom" not in profile.lower()):
                            one_person.append(np.unique(ptable["SCAN_PATH"]))
                            for path in select(sorted(np.unique(ptable["SCAN_PATH"]))):
                                #print(index, barcode, profile, path.split("_")[1])
                                path = os.path.join(PATH, path)
                                path = path.replace("\\", '/') 
                                one_person_images.append(PREPROCESS(np.asarray(Image.open(path))))
                                filenames.append(path)
                groups.append(one_person)
                IMAGES.append(one_person_images)
        except:
            continue
    return filenames, IMAGES, groups


# # Cписок пар изображений для каждого эксперимента


def pictures_num_to_differ(table):
    numbers = []
    for person in table:
        one_person_numbers = []
        count = -1
        for groups in person:
            count+=1
            for i in range(len(groups)-1):
                one_person_numbers.append((count,count+1))
                count+=1
        numbers.append(np.array(one_person_numbers))
    return numbers


# # Список вычетов одного изображения из другого (работает пока медленно)

def image_subtraction(images, numbers):
    subtractions = []
    all_subs = []
    k = 0
    for p_img, p_num in tqdm(zip(images, numbers)):
        k += 1
        person_subtractions = []
        for i in p_num:
            person_subtractions.append(abs(p_img[i[1]].astype('int16')-p_img[i[0]].astype('int16')))
        subtractions.append(person_subtractions)
        all_subs.extend(person_subtractions)
    return subtractions, all_subs

#Пайплайн

# # Считываем описание

description = pd.read_csv(DESCR, ",", decimal=".",
                          index_col=0, dayfirst=True,
                          encoding='utf-8')
description = description[COLUMNS]
description = add_result(description)
if len(PROFILE) > 0:
    description = description[description["Kiestra_CS_DESCRIPTION"].isin(PROFILE)]
print(description.head())
EXPERIMENT = ["Номер", "SCAN_PATH", "Kiestra_SCAN_NR", "Kiestra_CS_DESCRIPTION", "Kiestra_BARCODE"]
PATIENTS = description[EXPERIMENT].groupby("Номер")
# # Делим на Train и Test

# Все пациенты

patients = description[["Номер", "RESULT"]].groupby("Номер").mean()


ALL_NOGROWTH = len(patients[patients['RESULT']==False])
ALL_UNKNOWN = len(patients[patients['RESULT']==True])


# Делим на выборки

unknown_train, unknown_test = select(patients[patients["RESULT"]].index, ALL_UNKNOWN, train=TRAIN)
no_growth_train, no_growth_test = select(patients[~patients["RESULT"]].index, ALL_NOGROWTH, train=TRAIN)
print(len(unknown_train), len(unknown_test), TRAIN, ALL_UNKNOWN)
print(len(no_growth_train), len(no_growth_test), TRAIN, ALL_NOGROWTH)

#Получение изображений
_,images_ng, table_ng = read(no_growth_train, silent=False)
#print(len(images_ng))
for index, i in zip(range(len(images_ng)), images_ng):
    np.save("images_ng_blood_500_new/"+str(index), i)
#images_ng = np.load("images_ng.npy", allow_pickle=True)
#np.save("table_ng", table_ng)
_, images_un, table_un = read(unknown_train, silent=False)
#images_un = np.load("images_un.npy", allow_pickle=True)
for index, i in zip(range(len(images_un)), images_un):
    np.save("images_un_blood_500_new/"+str(index), i)
#print(len(table_ng))
#print(len(table_un))
#def getlen(table):
#    sum = 0
#    for i in table:
#        sum+=len(i)
#    return sum
#t = np.load("lol", allow_pickle=True)
#np.save("table_ng", table_un)
#Составление пар
del table_ng[234]
del table_un[291]
del table_un[288]
del table_un[131]
numbers_ng = pictures_num_to_differ(table_ng)
numbers_un = pictures_num_to_differ(table_un)

#print(len(table_ng))
#print(len(table_un))
#images_ng = []
#for i in tqdm(range(264)):
#    images_ng.append(np.load("images_ng_blood_200_new/"+str(i)+".npy"))
#    print(getlen(table_ng[i]), len(images_ng[i]))
    
#images_un = []
#for i in tqdm(range(453)):
#    images_un.append(np.load("images_un_blood_200_new/"+str(i)+".npy"))
#    print(len(table_un[i]), len(table_un[i]))

#Вычитание пар
per_subs_ng, all_subs_ng = image_subtraction(images_ng, numbers_ng)
per_subs_un, all_subs_un = image_subtraction(images_un, numbers_un)
#all_subs_ng = np.load("all_subs_ng_500.npy", allow_pickle=True)

np.save("all_subs_ng_blood_500_new", all_subs_ng)
np.save("all_subs_un_blood_500_new", all_subs_un)
#np.load("lol")
#all_subs_ng = np.load("all_subs_ng_blood_200_new.npy", allow_pickle=True)
#all_subs_un = np.load("all_subs_un_blood_200_new.npy", allow_pickle=True)
#print(len(all_subs_ng))
#print(len(all_subs_un))

#
print(len(all_subs_un))
all_subs_un_means = []
for i in all_subs_un:
    all_subs_un_means.append(i.mean())
all_subs_un = np.array(all_subs_un)    
all_subs_un_means = np.array(all_subs_un_means)    
all_subs_un = all_subs_un[all_subs_un_means>0.05]
print(len(all_subs_un))
all_subs_len = len(all_subs_un)+len(all_subs_ng)
FREQ = [len(all_subs_un)/all_subs_len,len(all_subs_ng)/all_subs_len]
class_weight = {0:len(all_subs_ng)/all_subs_len,
                1:len(all_subs_un)/all_subs_len}
from skimage.transform import rotate
from random import randint
from sklearn.utils import shuffle


X_data = []
for i in tqdm(all_subs_un):
    X_data.append(i)
for i in tqdm(all_subs_ng):
    X_data.append(i)
X_data = np.array(X_data)
y = [0]*len(all_subs_un) + [1]*len(all_subs_ng)
X_data, y = shuffle(X_data,y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_data, y, test_size=0.2, random_state=42)
#np.save("y_val_200_pr_ag6_XCEPTION_newloss",  y_val)
#np.load("lol")    
X_train_true = []
for i in tqdm(X_train):
    X_train_true.append(i)
    a = randint(1, 359)
    X_train_true.append(rotate(i,a,preserve_range = True).astype('uint8'))
    a = randint(1, 359)
    X_train_true.append(rotate(i,a,preserve_range = True).astype('uint8'))
    a = randint(1, 359)
    X_train_true.append(rotate(i,a,preserve_range = True).astype('uint8'))
y_train = np.repeat(y_train,4)
#X = []
#for i in tqdm(range(len(X_data))):
#    X.append(PREPROCESS(X_data[i]))
X_train_true, y_train = shuffle(X_train_true,y_train)
X_train = np.array(X_train_true)
np.save("X_train_200_XCEPTION", X_train)
np.save("y_train_200_XCEPTION",  y_train)
np.save("X_val_200_XCEPTION", X_val)
np.save("y_val_200_XCEPTION",  y_val)    
#X_train = np.array(X_train)
#X_data = np.array(X_data)
#X_train = np.load("X_train.npy", allow_pickle=True)
#y_train = np.load("y_train.npy", allow_pickle=True)
#X_train, y_train = shuffle(X_train, y_train)
#X_val = np.load("X_val.npy", allow_pickle=True)
#y_val = np.load("y_val.npy", allow_pickle=True)
#from skimage.color import rgb2gray

#X_train_gs = []
#for i in X_train:
#    X_train_gs.append(rgb2gray(i))
#X_val_gs = []
#for i in X_val:
#    X_val_gs.append(rgb2gray(i))
#X_train = np.reshape(X_train_gs, (len(X_train_gs), 200, 200, 1))
#X_val = np.reshape(X_val_gs, (len(X_val_gs), 200, 200, 1))

print(X_train.shape)
print(len(y_train))
print(X_val.shape)
print(len(y_val))
#from tensorflow.keras import regularizers

#from keras import layers
#from keras.callbacks import LearningRateScheduler

#learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=2, min_lr=0.00001)
learning_rate_reduction = LearningRateScheduler(lambda x: 1e-4 * 0.9 ** x)


datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=35,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    rescale=1./255,
    zoom_range=0.1)
datagen_valid = tf.keras.preprocessing.image.ImageDataGenerator()


def mywloss(y_true,y_pred):

    yc=tf.clip_by_value(y_pred,1e-15,1-1e-15)
    loss=-(tf.reduce_mean(tf.reduce_mean(y_true*tf.math.log(yc)/FREQ[1] + (1- y_true)*tf.math.log(1-yc)/FREQ[0], axis=0)))
    return loss



#datagen_train.fit(X_train)

#train = datagen_train.flow(np.array(X_train), y_train, batch_size=32)
#validation = datagen_valid.flow(np.array(X_val), y_val, batch_size=32)

base_model = tf.keras.applications.Xception(
    weights='imagenet', 
    input_shape=(500, 500, 3),
    include_top=False)

inputs = tf.keras.Input(shape=(500, 500, 3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(512, activation='relu', kernel_initializer='he_uniform')(x)
#x = layers.Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_pretrained = tf.keras.Model(inputs, outputs)

model_pretrained.compile(loss = 'binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print(model_pretrained.summary())




model = tf.keras.Sequential()
model.add(layers.Conv2D(32, kernel_size=(5,5), activation='relu', padding = 'same',kernel_initializer='he_uniform', input_shape=(200,200,3)))
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


model.compile(loss = 'binary_crossentropy',
              optimizer='adam',
             metrics=['accuracy'])
print(model.summary())
model_pretrained.fit(
        X_train, y_train,
        epochs=35,
        batch_size = 32,
        verbose = 2,
        validation_data = (X_val,y_val),
        class_weight=class_weight,
        callbacks = [learning_rate_reduction])
#model.fit(
#        X_train, y_train,
#        epochs=30,
#        batch_size = 16,
#        verbose = 2,
#        validation_data = (X_val,y_val),
#        callbacks = [learning_rate_reduction])
from sklearn.metrics import accuracy_score
#y_pred = np.round(model.predict(X_val))
y_pred = np.round(model_pretrained.predict(X_val))
X_err = []
for y_v, y_p, x in zip(y_val, y_pred, X_val):
    if(y_v != y_p):
        X_err.append(x)
np.save("X_err_200_XCEPTION", X_err)
np.save("y_pred_200_XCEPTION", y_pred)
print(accuracy_score(y_pred, y_val))
#print(accuracy_score(np.round(model.predict(X_val)), y_val))
#model.fit(
#        train,
#        steps_per_epoch = len(X_train) / 32,
#        epochs=30,
#        verbose = 1,
#        validation_data = validation,
#        validation_steps = len(X_val) / 32,
#        callbacks = [learning_rate_reduction])

