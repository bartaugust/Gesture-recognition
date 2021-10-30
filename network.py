import tensorflow as tf
from tensorflow.keras import models, layers
from parameters import *
import tensorflow_datasets as tfds
from tensorflow.keras import callbacks
from abc import abstractmethod
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def get_dataset():
    train_audio_path = tf.keras.utils.get_file(
        "imagenet",
        "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_test_v10102019.tar",
        cache_subdir='datasets\imagenet',
        extract=True)

    # dataset = tfds.load('imagenet2012')
    # return dataset


# def get_rnn(vocab_size):
#     model = models.Sequential()
#     # model.add(layers.Masking(input_shape=(MAX_FRAMES,)))
#     model.add(layers.GRU(16, return_sequences=True, input_shape=(MAX_FRAMES, NUM_FEATURES)))
#     model.add(layers.GRU(8))
#     model.add(layers.Dropout(0.4))
#     model.add(layers.Dense(8, activation='relu'))
#     model.add(layers.Dense(vocab_size, activation='softmax'))
#     model.compile(
#         loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
#     )
#
#     return model


class Network:
    def __init__(self):
        self.model = models.Model()
        self.history = callbacks.History()

    def fit_model(self, inputs, outputs):
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=ES_PATIENCE, min_delta=ES_MIN_DELTA)
        mc = ModelCheckpoint('best_model.hdf5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

        self.history = self.model.fit(inputs, outputs, validation_split=VAL_SIZE, epochs=EPOCHS, callbacks=[es, mc])

    def plot_results(self):
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.show()

    def evaluate(self, inputs, outputs):
        loss, accuracy = self.model.evaluate(inputs, outputs)
        print(f'test loss: {loss}')
        print(f'test accuracy: {accuracy}')
        return loss, accuracy

    def get_model(self):
        return self.model

    def get_history(self):
        return self.history


class ConvolutionalNetwork(Network):
    def __init__(self):
        super().__init__()
        input_layer = layers.Input(VID_SHAPE + (3,))

        conv_layer_1 = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        max_pool_layer_1 = layers.MaxPooling2D((2, 2))(conv_layer_1)

        flatted_layer = layers.Flatten()(max_pool_layer_1)
        dense_layer_1 = layers.Dense(NUM_FEATURES, activation='relu')(flatted_layer)

        output_layer = layers.Dense()(dense_layer_1)
        self.model = models.Model(input_layer, output_layer)


class RecurrentNetwork(Network):
    def __init__(self, vocab_size):
        super().__init__()
        frame_features_input = layers.Input((MAX_FRAMES, NUM_FEATURES))
        mask_input = layers.Input((MAX_FRAMES,), dtype="bool")

        # Refer to the following tutorial to understand the significance of using `mask`:
        # https://keras.io/api/layers/recurrent_layers/gru/
        x = layers.GRU(16, return_sequences=True)(
            frame_features_input, mask=mask_input
        )
        x = layers.GRU(8)(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(8, activation="relu")(x)
        output = layers.Dense(vocab_size, activation="softmax")(x)

        self.model = models.Model([frame_features_input, mask_input], output)

        self.model.compile(
            loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

# get_dataset()
