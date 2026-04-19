import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
import numpy as np
import math
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from matplotlib import pyplot as plt


class Sequential_Network:
    def __init__(
        self,
        input_shape,
        layers_number,
        neurons_number,
        num_classes,
        class_weight,
        verbose=0,
    ):
        self.input_shape = input_shape
        self.layers_number = layers_number
        self.neurons_number = neurons_number
        self.learning_rate = (
            0.5
            * (10 ** -(3 + ((layers_number) / 2.5)))
            * (0.7 ** (math.log2(neurons_number) - 5))
        )

        self.model = None
        self.num_classes = num_classes
        self.class_weight = class_weight
        self.verbose = verbose

    def build_model(self):
        middle_layers = self.get_middle_layers()
        self.model = keras.Sequential(
            [
                keras.layers.Input(shape=(self.input_shape,)),
                *middle_layers,
                keras.layers.Dense(self.num_classes, activation="softmax"),
            ]
        )
        print("Model built correctly...\n")
        print("Model summary: \n")
        self.model.summary()

    def compile_model(self):
        if self.model == None:
            raise Exception("Model not built yet. Build the model before compiling.")
        self.model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
        )

    def train(
        self,
        x_train,
        y_train,
        x_val,
        y_val,
        batch_size,
        epochs,
    ):
        if self.model == None:
            raise Exception(
                "Model not built yet. Build and compile the model before training."
            )

        early_stopping = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=15
        )
        # Define a learning rate schedule using ReduceLROnPlateau callback
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=(self.learning_rate / 30)
        )
        train_history = self.model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(x_val, y_val),
            class_weight=self.class_weight,
            verbose=self.verbose,
            callbacks=[early_stopping, reduce_lr],
        )

        return train_history

    def predict_probabilities(self, test_x_data):
        if self.model == None:
            raise Exception("Model not built yet. Build the model before compiling.")
        return self.model.predict(test_x_data)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model_from_file(file_path):
        return keras.saving.load_model(file_path)

    def make_prediction(self, x_test, y_test, categories):
        # 1. Get the raw probability distributions (e.g., [[0.1, 0.9], [0.8, 0.2]])
        probabilities = self.model.predict(x_test)

        # 2. Get the index of the highest probability for each row (e.g., [1, 0])
        prediction = np.argmax(probabilities, axis=1)

        # 3. If y_test is also One-Hot encoded, you MUST argmax it too
        # so that confusion_matrix is comparing "integers vs integers"
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_integers = np.argmax(y_test, axis=1)
        else:
            y_test_integers = y_test

        conf_matrix = pd.DataFrame(
            confusion_matrix(y_test_integers, prediction),
            index=categories,
            columns=categories,
        )

        return conf_matrix

    def get_performance_metrics(self, x_test, y_test):
        # 1. Get predictions (integers)
        probabilities = self.model.predict(x_test)
        y_pred = np.argmax(probabilities, axis=1)

        # 2. Convert One-Hot y_test back to integers
        y_true = np.argmax(y_test, axis=1)

        # 3. Print the report
        # This gives you Precision, Recall, and F1-score for EACH class
        print("\n--- Classification Report ---")
        print(classification_report(y_true, y_pred))

        # 4. Accuracy is still good to see
        acc = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {acc:.4f}")

    def get_middle_layers(self):
        model_layers = []
        for i in range(0, self.layers_number):
            model_layers.append(
                keras.layers.Dense(self.neurons_number, activation="relu")
            )
        return model_layers

    # Plot the history of training loss. kwargs contains the training parameter
    # I'm using (they can be useful when scanning)
    def plot_history(history, **kwargs):
        title_params = ""
        for key, value in kwargs.items():
            title_params = title_params + f"{key} = {value}, "

        plt.figure(1)
        plt.plot(history.history["loss"], color="green")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.title("Training with " + title_params)

        plt.figure(2)
        plt.title("Validation")
        plt.plot(history.history["val_loss"], color="green")
        plt.xlabel("epoch")
        plt.ylabel("Loss")
        plt.show()

    def scan_network_parameters(
        train_x_data,
        train_y_data,
        val_x_data,
        val_y_data,
        input_shape,
        layers_numbers,
        neurons_numbers,
        batch_sizes,
        num_classes,
        class_weight=None,
    ):
        min_loss = 100.0
        netw_opt_params = {}
        for layers in layers_numbers:
            for neurons in neurons_numbers:
                for batch_size in batch_sizes:
                    seq_netw = Sequential_Network(
                        input_shape=input_shape,
                        layers_number=layers,
                        neurons_number=neurons,
                        num_classes=num_classes,
                        class_weight=class_weight,
                    )
                    seq_netw.build_model()
                    seq_netw.compile_model()
                    train_history = seq_netw.train(
                        x_train=train_x_data,
                        y_train=train_y_data,
                        x_val=val_x_data,
                        y_val=val_y_data,
                        batch_size=batch_size,
                        epochs=200,
                    )

                    model_loss = (
                        sum(train_history.history["val_loss"][-3:]) / 3
                    )  # Averaged over the last three values

                    if model_loss < min_loss:
                        min_loss = model_loss
                        netw_opt_params = {
                            "layers": layers,
                            "neurons": neurons,
                            "batch_size": batch_size,
                        }

                    print(
                        "Validation loss for parameters layers, neurons, batch_size, learn_rate: "
                        + f"{layers}, {neurons}, {batch_size}, {seq_netw.learning_rate} = {model_loss}"
                    )
                    Sequential_Network.plot_history(
                        train_history,
                        layers=layers,
                        neurons=neurons,
                        batch_size=batch_size,
                    )
        return min_loss, netw_opt_params
