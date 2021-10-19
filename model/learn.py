import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
import data_preparation
import augmentation as aug
import parameters
import helper_functions as hf

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def get_cnn():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def get_rnn():
    model = models.Sequential()
    model.add(layers.GRU(16, return_sequences=True, input_shape=(IMG_SIZE, IMG_SIZE,)))
    model.add(layers.GRU(8))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(8, activation="relu"))
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def learn():
    pass


train_df, test_df = data_preparation.train_df, data_preparation.test_df

cnn_model = get_cnn()
rnn_model = get_rnn()
