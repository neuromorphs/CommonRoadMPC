import os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Dense
from tensorflow import keras
from sklearn import preprocessing
import joblib
import matplotlib.pyplot as plt
from globals import *
from nn_prediction.training.util import *
import json


FINETUNED_MODEL_NAME = "Fitting-Test"
FINETUNE_DATA_FILE = "finetune_data_vehicle_1.csv"


class NeuralNetworkFinetuner:
    def __init__(self, model_name=MODEL_NAME):

        print("Initializing nn_finetuner with model name {}".format(model_name))

        self.model_name = model_name

        # load model
        model_path = "nn_prediction/models/{}".format(self.model_name)
        scaler_x_path = "nn_prediction/models/{}/scaler_x.pkl".format(self.model_name)
        scaler_y_path = "nn_prediction/models/{}/scaler_y.pkl".format(self.model_name)
        nn_settings_path = "nn_prediction/models/{}/nn_settings.json".format(
            self.model_name
        )

        # chack if model is already trained
        if not os.path.isdir(model_path):
            print("Model {} does not exist. Please train first".format(self.model_name))
            exit()

        self.model = keras.models.load_model(model_path)
        self.scaler_x = joblib.load(scaler_x_path)
        self.scaler_y = joblib.load(scaler_y_path)
        with open(nn_settings_path, "r") as openfile:
            self.nn_settings = json.load(openfile)

        self.predict_delta = self.nn_settings["predict_delta"]
        self.normalize_data = self.nn_settings["normalize_data"]

        print("Model loaded...")

    def finetune_model(self):

        nn_settings = {
            "predict_delta": PREDICT_DELTA,
            "normalize_data": NORMALITE_DATA,
            "model_name": FINETUNED_MODEL_NAME,
        }

        # load the dataset
        # time, x1,x2,x3,x4,x5,x6,x7,u1,u2,x1n,x2n,x3n,x4n,x5n,x6n,x7n
        train_data = np.loadtxt(
            "nn_prediction/training/data/{}".format(FINETUNE_DATA_FILE), delimiter=","
        )

        # limit data for debugging
        # train_data = np.loadtxt('nn_prediction/training_data_small.csv', delimiter=',')

        print("train_data.shape", train_data.shape)

        # split into input (X) and output (y) variables
        # time, x1,x2,x3,x4,x5,x6,x7,u1,u2,x1n,x2n,x3n,x4n,x5n,x6n,x7n
        x = train_data[:, 1:10]
        y = train_data[:, 10:]

        # delta is the difference between state and next_state
        delta = y[:] - x[:, :7]

        # if we want to train the network on the state changes instead of the state, use this
        if PREDICT_DELTA:
            y = delta

        # Augmentation for lots of lots of data
        # x, y = augment_data(x,y)

        # Normalize data
        scaler_x = preprocessing.MinMaxScaler().fit(x)
        scaler_y = preprocessing.MinMaxScaler().fit(y)
        if NORMALITE_DATA:
            x = scaler_x.transform(x)
            y = scaler_y.transform(y)

        print("Fine tuning model with new data")

        # Freze first layers

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in self.model.layers[:-2]:
            layer.trainable =  False


        self.model.summary()
            # fit
        history = self.model.fit(
            x,
            y,
            epochs=NUMBER_OF_EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            initial_epoch=self.model.history.epoch[-1],
            validation_split=0.1,
        )

        # Save the fitted model
        model_path = "nn_prediction/models/{}".format(FINETUNED_MODEL_NAME)
        scaler_x_path = "nn_prediction/models/{}/scaler_x.pkl".format(FINETUNED_MODEL_NAME)
        scaler_y_path = "nn_prediction/models/{}/scaler_y.pkl".format(FINETUNED_MODEL_NAME)
        nn_settings_path = "nn_prediction/models/{}/nn_settings.json".format(FINETUNED_MODEL_NAME)


        self.model.save(model_path)
        joblib.dump(scaler_x, scaler_x_path)
        joblib.dump(scaler_y, scaler_y_path)
        with open(nn_settings_path, "w") as outfile:
            outfile.write(json.dumps(nn_settings))

        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("nn_prediction/models/{}/accuracy_curve".format(FINETUNED_MODEL_NAME))

        plt.clf()
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.yscale("log")
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "val"], loc="upper left")
        plt.savefig("nn_prediction/models/{}/loss_curve".format(FINETUNED_MODEL_NAME))

        # Evaluate
        _, accuracy = self.model.evaluate(x, y)
        print("Accuracy: %.2f" % (accuracy * 100))




if __name__ == '__main__':

    tuner = NeuralNetworkFinetuner(model_name="Dense-128-128-128-128-uniform-20")
    tuner.finetune_model()
