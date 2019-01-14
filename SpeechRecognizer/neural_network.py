import tensorflow as tf
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras import utils
import matplotlib.pyplot as plt
import os
cwd = os.path.abspath(os.path.join("", os.pardir))
catdir = cwd+"\\Categories.txt"
file = open(catdir, "r")
CATEGORIES = file.read().replace('\n', ',').split(',')
file.close()


class DNN:
    def __init__(self, epochs=135, batch_size=16, validation_split=0.2, categories=CATEGORIES):

        pickle_in = open("X.pickle", "rb")
        x_train = pickle.load(pickle_in)
        self.x_train = np.array(x_train)
        #self.x_train = tf.keras.utils.normalize(self.x_train)
        pickle_in = open("Y.pickle", "rb")
        y_train = pickle.load(pickle_in)
        y_train = np.array(y_train)
        lb = LabelEncoder()
        self.y_train = utils.to_categorical(lb.fit_transform(y_train))
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.categories = categories

    def train(self):

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=256, activation='tanh', input_dim=self.x_train.shape[1]))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.Dense(units=256, activation='tanh'))
        model.add(tf.keras.layers.Dropout(0.1))
        model.add(tf.keras.layers.Dense(units=len(self.categories), activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adamax')
        history = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs,
                            validation_split=self.validation_split)
        model.save("Model.h5")
        return history


class RNN:
    def __init__(self, epochs=50, batch_size=16, validation_split=0.2, categories=CATEGORIES):

        pickle_in = open("X.pickle", "rb")
        x_train = pickle.load(pickle_in)
        self.x_train = np.array(x_train)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 87, 20)
        self.x_train = tf.keras.utils.normalize(self.x_train)
        pickle_in = open("Y.pickle", "rb")
        y_train = pickle.load(pickle_in)
        y_train = np.array(y_train)
        lb = LabelEncoder()
        self.y_train = utils.to_categorical(lb.fit_transform(y_train))
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.categories = categories

    def train(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=256, input_shape=self.x_train.shape[1:],return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        model.add(tf.keras.layers.LSTM(units=256))
        model.add(tf.keras.layers.Dropout(0.2))
        adam = tf.keras.optimizers.Adam(lr=0.001)
        model.add(tf.keras.layers.Dense(units=len(self.categories), activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
        print(model.summary())
        history = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs
                            , validation_split=self.validation_split
                            , verbose=1)
        model.save("ModelRNN.h5")
        return history


def plot(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

