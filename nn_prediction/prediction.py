import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
from globals import *

from nn_prediction.training.train import train_network


#load model
model_path = 'nn_prediction/models/{}'.format(MODEL_NAME)
scaler_x_path = 'nn_prediction/models/{}/scaler_x.pkl'.format(MODEL_NAME)
scaler_y_path = 'nn_prediction/models/{}/scaler_y.pkl'.format(MODEL_NAME)

#chack if model is already trained
if not os.path.isdir(model_path):
    print("Model {} does not exist. Starting training...".format(MODEL_NAME))
    train_network()

model = keras.models.load_model(model_path)
scaler_x = joblib.load(scaler_x_path) 
scaler_y = joblib.load(scaler_y_path) 
    


def predict_next_state(state, control_input):

    state_and_control = np.append(state, control_input)

    # Normalize input
    state_and_control_normalized = scaler_x.transform([state_and_control])

    # Predict
    predictions_normalized = model.predict(state_and_control_normalized)

    # Denormalize results
    prediction = scaler_y.inverse_transform(predictions_normalized)[0]

    return prediction






if __name__ == '__main__':

    next_state = predict_next_state([41.900000000000006,13.600000000000001,0.0,7.0,0.0,0.0,0.0], [0.06644491781040185,-0.40862345615368234])
    print("Next_state", next_state)
