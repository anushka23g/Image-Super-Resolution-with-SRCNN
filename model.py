import os
import numpy as np
from keras.models import Sequential, model_from_json
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam

from keras.layers.core import Activation

class SRCNN:
    def __init__(self, image_size, c_dim, is_training, learning_rate=1e-4, batch_size=128, epochs=10000):
        self.image_size = image_size
        self.c_dim = c_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.is_training = is_training
        if self.is_training:
            self.model = self.build_model()
        else:
            self.model = self.load()    
    
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64,9,padding='same',input_shape=(self.image_size,self.image_size,self.c_dim)))
        model.add(Activation('relu'))
        model.add(Conv2D(32,1,padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(self.c_dim,5,padding='same'))
        optimizer = Adam(lr=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
        return model
    
    def train(self, X_train, Y_train):
        history = self.model.fit(X_train, Y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1, validation_split=0.1)
        if self.is_training:
            self.save()
        return history
    
    def process(self, input):
        predicted = self.model.predict(input)
        return predicted
    
    def load(self):
        weight_filename = 'srcnn_weight.hdf5'
        model = self.build_model()
        model.load_weights(os.path.join('./model/',weight_filename))
        return model

    def save(self):
        json_string = self.model.to_json()
        open(os.path.join('./model/','srcnn_model.json'),'w').write(json_string)
        self.model.save_weights(os.path.join('./model/','srcnn_weight.hdf5'))
        return json_string
