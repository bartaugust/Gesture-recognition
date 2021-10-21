import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
import data_preparation as dp
import augmentation as aug
from parameters import *
import helper_functions as hf
import numpy as np


def get_cnn(preset='inception_v3'):
    if preset == 'inception_v3':
        feature_extractor = keras.applications.InceptionV3(
            weights="imagenet",
            include_top=False,
            pooling="avg",
            input_shape=VID_SHAPE + (3,),
        )
        preprocess_input = keras.applications.inception_v3.preprocess_input

        inputs = keras.Input(VID_SHAPE + (3,))
        preprocessed = preprocess_input(inputs)

        outputs = feature_extractor(preprocessed)
        return keras.Model(inputs, outputs, name="feature_extractor")
    else:
        return None


def get_rnn():
    model = models.Sequential()
    model.add(layers.GRU(16, return_sequences=True, input_shape=VID_SHAPE))
    model.add(layers.GRU(8))
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(8, activation="relu"))
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    return model


def run_cnn(cnn, df, vocab):
    num_samples = len(df)
    all_features = np.zeros(
        shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
    )
    all_masks = []

    vocab_layer = tf.keras.layers.StringLookup(vocabulary=vocab)
    labels = [vocab_layer(df["tag"].values)]

    for idx, row in df.iterrows():
        frames = hf.load_video('data\\' + row['tag'] + '\\' + row['video_name'], max_frames=FRAMES,
                               resize=VID_SHAPE)
        video_features = np.zeros(
            shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )
        for i in range(min(MAX_SEQ_LENGTH, frames.shape[0])):
            frame = frames[i]
            features = cnn.predict(frame[None, ...])
            video_features[:, i, :] = features
        all_features[idx,] = video_features
    return all_features, all_masks, labels


def learn():
    pass


train_df, test_df = dp.prepare_dataframes()

cnn_model = get_cnn()
rnn_model = get_rnn()

train_features, train_masks, train_labels = run_cnn(cnn_model, train_df, dp.vocab)
test_features, test_masks, test_labels = run_cnn(cnn_model, test_df, dp.vocab)
