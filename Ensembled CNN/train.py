
# Import Libraries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras.models import Sequential
from keras.layers import Conv2D, Lambda, MaxPooling2D # convolution layers
from keras.layers import Dense, Dropout, Flatten # core layers
from tensorflow.keras.layers import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.utils.np_utils import to_categorical



TRAIN_DATA_LOC = './data/train.csv'
TRAIN_LABEL_LOC = './data/train_result.csv'

def load_and_preprocess(train_data_loc = TRAIN_DATA_LOC, train_label_loc = TRAIN_LABEL_LOC):
    X_train = pd.read_csv(train_data_loc)
    y = pd.read_csv(train_label_loc, index_col = 0).reset_index(drop=True)
    
    # Remove irrelevant columns
    X_train = X_train.iloc[:, :-1]

    # Transforming into a numpy array
    X_train = X_train.values.reshape((-1,28,56,1))

    # One-Hot Encoding the labels
    y = to_categorical(y)

    return X_train, y

def build_model(X_train):
    # BUILD CONVOLUTIONAL NEURAL NETWORKS
    nets = 15
    model = [0] *nets
    for j in range(nets):
        model[j]=Sequential()
        model[j].add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=X_train.shape[1:]))
        model[j].add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
        model[j].add(MaxPooling2D(pool_size=(2,2)))
        model[j].add(BatchNormalization())

        model[j].add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model[j].add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
        model[j].add(MaxPooling2D(pool_size=(2,2)))
        model[j].add(BatchNormalization())    

        model[j].add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
        model[j].add(MaxPooling2D(pool_size=(2,2)))
        model[j].add(BatchNormalization())

        model[j].add(Flatten())
        model[j].add(Dense(512,activation="relu"))

        model[j].add(Dense(19,activation="softmax"))

        model[j].compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return nets, model

def train():

    X_train, y = load_and_preprocess(TRAIN_DATA_LOC, TRAIN_LABEL_LOC)

    # CREATE MORE IMAGES VIA DATA AUGMENTATION
    datagen = ImageDataGenerator(
            rotation_range=10,  
            zoom_range = 0.10,  
            width_shift_range=0.1, 
            height_shift_range=0.1)

    
    nets, model = build_model(X_train)


    history = [0] * nets
    epochs = 100
    for j in range(nets):
            history[j] = model[j].fit(datagen.flow(X_train,y, batch_size=128),
                                                epochs = epochs)
            print("CNN {0:d}: Epochs={1:d}, Train accuracy={2:.5f}".format(j+1,epochs,max(history[j].history['accuracy'])))
        
    
    for j in range(nets):
        model[j].save(f'./saved models/ensemble-c5/{j}')

if __name__ == '__main__':
    train()