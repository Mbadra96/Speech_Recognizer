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
        #LabelEncoder encode labels with a value between 0 and n_classes-1
        # where n is the number of distinct labels.
        # If a label repeats it assigns the same value to as assigned earlier.
        #The problem here is since there are different numbers in the same column,
        #  the model will misunderstand the data to be in some kind of order, 0 < 1 <2.
        #hotkey
        #beydeek ones w zeros w blaah blaah blahhh blaah
        self.y_train = utils.to_categorical(lb.fit_transform(y_train))
        #Converts a class vector (integers) to binary class matrix.
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

#accuracy no3en sa
#traininging w validation
#hyperpapmaters el homa  epochs wi activation funciton batch size
#validation accuracy bey2sar validation split el betbaselha data lw 2alleet validaiton split small acurracy bet2al

class RNN:
    def __init__(self, epochs=50, batch_size=16, validation_split=0.2, categories=CATEGORIES):

        pickle_in = open("X.pickle", "rb")
        x_train = pickle.load(pickle_in)
        self.x_train = np.array(x_train)
        self.x_train = self.x_train.reshape(self.x_train.shape[0], 130, 20)
        #x.shape beydelek length of the zeor dimension
        #130
        #20
        #(sample rate/chunck)*seconds
        self.x_train = tf.keras.utils.normalize(self.x_train)
        pickle_in = open("Y.pickle", "rb")
        y_train = pickle.load(pickle_in)
        y_train = np.array(y_train)
        lb = LabelEncoder()
        self.y_train = utils.to_categorical(lb.fit_transform(y_train))
        self.epochs = epochs ## feedforward plus back propagation
        self.batch_size = batch_size #number of training examples utilized in one iteration
        self.validation_split = validation_split
        self.categories = categories

    def train(self):
        #.add to add a new layer
        #The model needs to know what input shape it should expect.
        # For this reason, the first layer in a Sequential model
        # (and only the first, because following layers can do automatic shape inference)
        #  needs to receive information about its input shape
        #input_shape

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.LSTM(units=256, input_shape=self.x_train.shape[1:],return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        # avoid over fitting
        #dropout consists in randomly setting a fraction rate of input units to 0 at each update
        #during training time preventing overfitting

        model.add(tf.keras.layers.LSTM(units=256))
        model.add(tf.keras.layers.Dropout(0.2))
        adam = tf.keras.optimizers.Adam(lr=0.001)
        model.add(tf.keras.layers.Dense(units=len(self.categories), activation='softmax'))
        model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)
        #Configures the model for training.
        #metric 3ayz a evaluate eh during trainning and testing
        print(model.summary())
        history = model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs
                            , validation_split=self.validation_split
                            , verbose=1)
        #Trains the model for a given number of epochs (iterations on a dataset).
        #fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
        #x nump array of training data.
        #y numpy array of labels.
        #Number of samples per gradient update
        #epoch number of epoch to train the model
        #verbose verbose=1 will show you an animated progress bar like this:
        # Its History.history attribute is a record of training loss values
        # and metrics values at successive epochs,
        # as well as validation loss values and validation metrics values (if applicable).
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

