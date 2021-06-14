from locale import normalize
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn import preprocessing
import matplotlib.pyplot as plt
from globals import *




# load the dataset
#time, x1,x2,x3,x4,x5,x6,x7,u1,u2,x1n,x2n,x3n,x4n,x5n,x6n,x7n
train_data = np.loadtxt('NN_Prediction/training_data.csv', delimiter=',')
print(train_data.shape)

# limit data for debugging
# train_data  = train_data[:1000]


# split into input (X) and output (y) variables
x = train_data[:,1:10]
y = train_data[:,10:]

# print("x", x.shape)
# print(x[0])
# print("y", y.shape)
# print(y[0])
# exit()

# Normalize
scaler_x = preprocessing.MinMaxScaler().fit(x)
scaler_y = preprocessing.MinMaxScaler().fit(y)
x = scaler_x.transform(x)
y = scaler_y.transform(y)



# keras model
model = Sequential()
model.add(Dense(128, input_dim=9, activation='tanh'))
model.add(Dense(256, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(7, activation='tanh'))


if TRAIN_NEW_MODEL:
    # compile the keras model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x, y, epochs=5, batch_size=10, shuffle=True, validation_split=0.1)
    model.save('nn_prediction/keras_model.km')

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

    #Evaluate
    _, accuracy = model.evaluate(x, y)
    print('Accuracy: %.2f' % (accuracy*100))
else:
    model = keras.models.load_model('nn_prediction/keras_model.km')






#Prediction

x_test = [
    [41.900000000000006,13.600000000000001,0.0,7.0,0.0,0.0,0.0, 0.06644491781040185,-0.40862345615368234],

]
#True value = 49.81352121241394,14.578289513486826,0.19588448877551679,7.069136403019563,0.2656700131292898,0.5242158977628238,0.08863795065628435
x_test = scaler_x.transform(x_test)
predictions = model.predict(x_test)

#Denormalize results
predictions = scaler_y.inverse_transform(predictions)


print("predictions", predictions)






def predict_next_state(state, control_input):

    state_and_control = np.append(state, control_input)
    print("state_and_control", state_and_control)

    state_and_control_normalized = scaler_x.transform([state_and_control])
    predictions_normalized = model.predict(state_and_control_normalized)

    #Denormalize results
    prediction = scaler_y.inverse_transform(predictions_normalized)[0]

    return prediction


# predict_next_state([48.52940997965647,14.184236643335186,0.16841573274034935,7.076115033187338,0.17417732213439957,0.43026457546845215,0.07354705438652182], [0.34065824691511365,0.1916452940425778])