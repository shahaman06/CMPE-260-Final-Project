import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  # kill warning about tensorflow
import tensorflow as tf
import numpy as np
import sys

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model
from constants import *


class TrainModel:
    def __init__(self):
        self.input_dim = N_STATES
        self.output_dim = N_ACTIONS
        self.batch_size = TRAIN_BATCH_SIZE
        self.lr = TRAIN_LR

        inputs = keras.Input(shape=(N_STATES,))
        x = layers.Dense(TRAIN_W_LAYERS, activation='relu')(inputs)
        for _ in range(TRAIN_N_LAYERS):
            x = layers.Dense(TRAIN_W_LAYERS, activation='relu')(x)
        outputs = layers.Dense(N_ACTIONS, activation='linear')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='my_model')
        model.compile(loss=losses.mean_squared_error, optimizer=Adam(lr=TRAIN_LR))
        self._model = model

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, N_STATES])
        return self._model.predict(state)


    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        return self._model.predict(states)


    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        self._model.fit(states, q_sa, epochs=1, verbose=0)


    def save_model(self, path):
        """
        Save the current model in the folder as h5 file and a model architecture summary as png
        """
        self._model.save(os.path.join(path, 'trained_model.h5'))
        plot_model(self._model, to_file=os.path.join(path, 'model_structure.png'), show_shapes=True, show_layer_names=True)

class TestModel:
    def __init__(self, input_dim, model_path):
        N_STATES = input_dim
        self._model = self._load_my_model(model_path)


    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.h5')
        
        if os.path.isfile(model_file_path):
            loaded_model = load_model(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")


    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = np.reshape(state, [1, N_STATES])
        return self._model.predict(state)


    @property
    def input_dim(self):
        return N_STATES