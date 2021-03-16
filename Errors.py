#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import listdir
from PIL import Image
from skimage.draw import circle
from skimage import transform, filters, exposure, feature
from sklearn.model_selection import train_test_split
import io
from skimage.color import rgb2gray


# # Chrom

# In[31]:


X = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\X_err_chr_all_train_one_valid.npy")
y = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\y_pred_chr_all_train_one_valid.npy")
y = y.reshape(len(y))


# In[39]:


print(f"Всего ошибок: {len(y)} из {round(len(y)/(1-0.987))}")


# ### 0 - выросло

# ### 1 - не выросло

# ### Модель предсказала 1, а нужно было 0

# In[33]:


print(f"Таких примеров: {sum(y)}\n")
for i, target in zip(X, y):
    if(target == 1):
        print(f"Model target: {target}")
        print(f"True target: {1-target}")
        plt.figure(figsize=(10,10))
        plt.imshow(i)
        plt.show()


# ### Модель предсказала 0, а нужно было 1

# In[34]:


print(f"Таких примеров: {len(y) - sum(y)}\n")
for i, target in zip(X, y):
    if(target == 0):
        print(f"Model target: {target}")
        print(f"True target: {1-target}")
        plt.figure(figsize=(10,10))
        plt.imshow(i)
        plt.show()


# # Blood

# In[36]:


X = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\X_err_blood_all_train_one_valid.npy")
y = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\y_pred_blood_all_train_one_valid.npy")
y = y.reshape(len(y))


# In[37]:


print(f"Всего ошибок: {len(y)} из {round(len(y)/(1-0.975))}")


# ### Модель предсказала 1, а нужно было 0

# In[24]:


print(f"Таких примеров: {sum(y)}\n")
for i, target in zip(X, y):
    if(target == 1):
        print(f"Model target: {target}")
        print(f"True target: {1-target}")
        plt.figure(figsize=(10,10))
        plt.imshow(i)
        plt.show()


# ### Модель предсказала 0, а нужно было 1

# In[38]:


print(f"Таких примеров: {len(y) - sum(y)}\n")
for i, target in zip(X, y):
    if(target == 0):
        print(f"Model target: {target}")
        print(f"True target: {1-target}")
        plt.figure(figsize=(10,10))
        plt.imshow(i)
        plt.show()


# # Изображения до отметки выросшей (Октябрь)

# ## Chrom

# In[47]:


X = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\X_err_500_XCEPTION_chr_pre_oct.npy")
y = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\y_pred_500_XCEPTION_chr_pre_oct.npy")
y = y.reshape(len(y))


# In[49]:


print(f"Таких примеров: {sum(y)}\n")
for i, target in zip(X, y):
    if(target == 1):
        print(f"Model target: {target}")
        print(f"True target: {1-target}")
        plt.figure(figsize=(10,10))
        plt.imshow(i)
        plt.show()


# ## Blood

# In[50]:


X = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\X_err_500_XCEPTION_blood_pre_oct.npy")
y = np.load("D:\\Medicine2020\\Kiestra\\NALETOV_IMG_2020\\new_wave\\y_pred_500_XCEPTION_blood_pre_oct.npy")
y = y.reshape(len(y))


# In[51]:


print(f"Таких примеров: {sum(y)}\n")
for i, target in zip(X, y):
    if(target == 1):
        print(f"Model target: {target}")
        print(f"True target: {1-target}")
        plt.figure(figsize=(10,10))
        plt.imshow(i)
        plt.show()


# In[ ]:




