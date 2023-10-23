import flwr as fl
import tensorflow as tf
import os
import time
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, x_train, y_train, x_test, y_test):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(self.x_train, self.y_train, epochs=10, batch_size=32, verbose=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": float(accuracy)}

def gen_dataset(path):
    data = np.load(path)
    x_train, y_train = data['train_images'], data['train_labels']
    x_test, y_test = data['test_images'], data['test_labels']
    num_classes = max(len(np.unique(y_train)), len(np.unique(y_test)))
    input_shape = x_train.shape[1:]
    return x_train, y_train, x_test, y_test, num_classes, input_shape

HOST = os.getenv("SERVERHOST")

dbs = [
    "./client/MedMNIST/bloodmnist.npz",
    "./client/MedMNIST/breastmnist.npz",
    "./client/MedMNIST/chestmnist.npz",
    "./client/MedMNIST/dermamnist.npz",
    "./client/MedMNIST/octmnist.npz",
    "./client/MedMNIST/organamnist.npz",
    "./client/MedMNIST/organcmnist.npz",
    "./client/MedMNIST/organsmnist.npz",
    "./client/MedMNIST/pathmnist.npz",
    "./client/MedMNIST/pneumoniamnist.npz",
    "./client/MedMNIST/retinamnist.npz",
    "./client/MedMNIST/tissuemnist.npz",
]

for DB_NAME in dbs:

    x_train, y_train, x_test, y_test, num_classes, input_shape = gen_dataset(DB_NAME)

    def get_model():
        inputs = tf.keras.Input(input_shape)
        model = tf.keras.applications.resnet.ResNet50(input_tensor=inputs, classes=num_classes, weights=None)
        return tf.keras.Model(inputs, model.output)

    model = get_model()

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    print(f"Starting client for {DB_NAME}\n\n")
    fl.client.start_numpy_client(server_address=f"{HOST}:8080", client=FlowerClient(x_train, y_train, x_test, y_test))
    print("------------------------------------\n\n")
    time.sleep(4)