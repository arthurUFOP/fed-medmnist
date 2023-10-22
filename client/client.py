import flwr as fl
import tensorflow as tf
import os
import numpy as np

def gen_dataset(path):
    data = np.load(path)
    x_train, y_train = data['train_images'], data['train_labels']
    x_test, y_test = data['test_images'], data['test_labels']
    num_classes = max(len(np.unique(y_train)), len(np.unique(y_test)))
    input_shape = x_train.shape[1:]
    return x_train, y_train, x_test, y_test, num_classes, input_shape

HOST = os.getenv("SERVERHOST")
DB_NAME = os.getenv("DB_NAME")
x_train, y_train, x_test, y_test, num_classes, input_shape = gen_dataset(os.path.join('MedMNIST', DB_NAME + ".npz"))

def get_model():
    inputs = tf.keras.Input(input_shape)
    model = tf.keras.applications.efficientnet.EfficientNetB0(input_tensor=inputs, classes=num_classes, weights=None)
    return tf.keras.Model(inputs, model.output)

model = get_model()

model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=1)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": float(accuracy)}

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

fl.client.start_numpy_client(server_address=f"{HOST}:8080", client=CifarClient())