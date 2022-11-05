
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
    model = Sequential()

    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())

    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())    

    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
        
    model.add(Flatten())
    model.add(Dense(512,activation="relu"))
        
    model.add(Dense(19,activation="softmax"))
        
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return model

def train():
    X_train, y = load_and_preprocess(TRAIN_DATA_LOC, TRAIN_LABEL_LOC)
    model = build_model(X_train)
    model.fit(X_train, y, batch_size=128, epochs=100)
    model.save('./saved models/cnn-c-5-100-iter')

if __name__ == '__main__':
    train()