import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
from globals import *
from nn_prediction.training.util import *


def train_network():

    # load the dataset
    #time, x1,x2,x3,x4,x5,x6,x7,u1,u2,x1n,x2n,x3n,x4n,x5n,x6n,x7n
    train_data = np.loadtxt('nn_prediction/training_data_large.csv', delimiter=',')

    # limit data for debugging
    # train_data = np.loadtxt('nn_prediction/training_data.csv', delimiter=',')
    # train_data  = train_data[:10]

    print("shape of trian data:", train_data.shape)



    # split into input (X) and output (y) variables
    x = train_data[:,1:10]
    y = train_data[:,10:]

    #delta is the difference between state and next_state
    delta =  y[:] - x[:,:7]

    #if we want to train the network on the state changes instead of the state, use this
    if PREDICT_DELTA:
        y = delta

    # print(y[0])
    # exit()

    # x, y = augument_data(x,y)

    # print("x.shape", x.shape)
    # print("y.shape", y.shape)

    # print("x_augumented",x[::10,0])
    # print("x_augumented",x[::10,1])
    # print("y_augumented",y.shape)
    # exit()


    # Normalize data
    scaler_x = preprocessing.MinMaxScaler().fit(x)
    scaler_y = preprocessing.MinMaxScaler().fit(y)
    x = scaler_x.transform(x)
    y = scaler_y.transform(y)


    # keras model
    model = Sequential()
    model.add(Dense(128, input_dim=9, activation='tanh'))
    model.add(Dense(256, activation='tanh'))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(7, activation='tanh'))


    # compile
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    # fit
    history = model.fit(x, y, epochs=NUMBER_OF_EPOCHS, batch_size=BATCH_SIZE, shuffle=True, validation_split=0.1)

    # Save model and normalization constants
    model_path = 'nn_prediction/models/{}'.format(MODEL_NAME)
    scaler_x_path = 'nn_prediction/models/{}/scaler_x.pkl'.format(MODEL_NAME)
    scaler_y_path = 'nn_prediction/models/{}/scaler_y.pkl'.format(MODEL_NAME)
    model.save(model_path)
    joblib.dump(scaler_x, scaler_x_path) 
    joblib.dump(scaler_y, scaler_y_path) 

    # Plot results
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    #Evaluate
    # _, accuracy = model.evaluate(x, y)
    # print('Accuracy: %.2f' % (accuracy*100))






if __name__ == '__main__':

    train_network()
