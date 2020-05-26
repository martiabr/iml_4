#!/usr/bin/env python
# coding: utf-8

# Imports:

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import os

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model


# Load images:

# In[25]:


folder_path = 'food/'
aim_folder_path = 'foodpreprocessed/'

img_width, img_height = 224, 224

# load all images into a list
file_list = sorted(os.listdir(folder_path))
images = []
names = []
i = 0
for img in file_list:
    if img == '.DS_Store':  # ignore stupid fookin mac file that wont go away
        continue
    thisname = img
    names.append(img)
    img = os.path.join(folder_path, img)
    img = image.load_img(img, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    np.save(aim_folder_path+thisname, img)
    images.append(img)
    i = i + 1

# stack up images list to pass for prediction
images = np.vstack(images)


# Load triplets:

# In[7]:


df_triplets = pd.read_csv('train_triplets.txt', sep=" ", header=None)
print(df_triplets)


# Train-validation split:

# In[8]:


df_triplets_train = pd.DataFrame(columns = [0, 1, 2])
df_triplets_val = pd.DataFrame(columns = [0, 1, 2])

N = 325
for idx, row in df_triplets.iterrows():
  if idx < N:  # first N rows go in train
    df_triplets_train = df_triplets_train.append(row)
  elif all(x not in df_triplets_train.values for x in [row[0], row[1], row[2]]):  # else if images not in train add to val
    df_triplets_val = df_triplets_val.append(row)
  elif all(x not in df_triplets_val.values for x in [row[0], row[1], row[2]]):  # else if images not in val add to train
    df_triplets_train = df_triplets_train.append(row)
  # else we discard triplet
  
df_triplets_train = df_triplets_train.reset_index(drop=True)
df_triplets_val = df_triplets_val.reset_index(drop=True)

print(df_triplets_train)
print(df_triplets_val)


# Swap every other element so that 0/1 labels are balanced:

# In[9]:


for i, row in df_triplets_train.iterrows():
  if i % 2 == 1:
    temp = row[1]
    df_triplets_train.at[i,1] = row[2]
    df_triplets_train.at[i,2] = temp
print(df_triplets_train)

for i, row in df_triplets_val.iterrows():
  if i % 2 == 1:
    temp = row[1]
    df_triplets_val.at[i,1] = row[2]
    df_triplets_val.at[i,2] = temp
print(df_triplets_val)


# Create labels:

# In[10]:


y_train = np.empty((df_triplets_train.shape[0], 1))
y_train[::2] = 1
y_train[1::2] = 0
print(y_train)

y_val = np.empty((df_triplets_val.shape[0], 1))
y_val[::2] = 1
y_val[1::2] = 0
print(y_val)


# Transform triplets of image indexes --> triplets of images:

# In[26]:


# TODO: do this transform
train_0 = []
train_1 = []
train_2 = []
test_0 = []
test_1 = []
test_2 = []
print(df_triplets_train[0])
print()

for imID in df_triplets_train[0]:
    train_0.append(np.load(aim_folder_path+str(imID).split('.')[0].zfill(5)+'.npy'))
for imID in df_triplets_train[1]:
    train_1.append(np.load(aim_folder_path+str(imID).split('.')[0].zfill(5)+'.npy'))
for imID in df_triplets_train[2]:
    train_2.append(np.load(aim_folder_path+str(imID).split('.')[0].zfill(5)+'.npy'))

for imID in df_triplets_test[0]:
    test_0.append(np.load(aim_folder_path+str(imID).split('.')[0].zfill(5)+'.npy'))
for imID in df_triplets_test[1]:
    test_1.append(np.load(aim_folder_path+str(imID).split('.')[0].zfill(5)+'.npy'))
for imID in df_triplets_test[2]:
    test_2.append(np.load(aim_folder_path+str(imID).split('.')[0].zfill(5)+'.npy'))
    
print(train_0)


# Setup model for transfer learning:

# In[11]:


model1 = ResNet50(weights='imagenet', include_top=False)
model2 = ResNet50(weights='imagenet', include_top=False)
model3 = ResNet50(weights='imagenet', include_top=False)

#  Make sure all layer names are unique (otherwise it gets upset) and freeze all pre-trained layers:
for layer in model1.layers:
  layer.trainable = False
  layer._name = layer._name + str("_1")
for layer in model2.layers:
  layer.trainable = False
  layer._name = layer._name + str("_2")
for layer in model3.layers:
  layer.trainable = False
  layer._name = layer._name + str("_3")

out1 = model1.output
out2 = model2.output
out3 = model3.output

x = layers.concatenate([out1, out2, out3])

out = layers.Dense(1, activation='softmax')(x)

model = Model(inputs=[model1.input,model2.input,model3.input], outputs=[out])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


# Train model:

# In[17]:


#TODO: train the model
print(train_0)
train_0 = np.vstack(train_0)
train_1 = np.vstack(train_1)
train_2 = np.vstack(train_2)
test_0 = np.vstack(test_0)
test_1 = np.vstack(test_1)
test_2 = np.vstack(test_2)

model.fit({train_0,train_1,train_2}, y_train,validation_data=({test_0,test_1,test_2}, y_val), epochs=5, batch_size=64)


# In[ ]:




