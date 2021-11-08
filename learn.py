import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
import data_preparation as dp
from parameters import *
import helper_functions as hf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import network
import augmentation as aug

train_data, test_data = dp.prepare_dataframes()


# %%

def get_cnn(preset=''):
    if preset == 'inception_v3':
        feature_extractor = keras.applications.InceptionV3(
            weights='imagenet',
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
        cnn = network.ConvolutionalNetwork()
    return cnn.get_model_without_output()


def run_cnn(cnn, df, vocab_layer):
    num_samples = len(df)
    all_features = np.zeros(
        shape=(num_samples * NUM_AUG, MAX_FRAMES, NUM_FEATURES), dtype="float32"
    )
    all_masks = np.zeros(shape=(num_samples * NUM_AUG, MAX_FRAMES), dtype="bool")

    labels = np.zeros(shape=(num_samples * NUM_AUG, 1))

    for idx, row in df.iterrows():

        frames = hf.load_video('data\\' + row['tag'] + '\\' + row['video_name'], max_frames=FRAMES,
                               resize=VID_SHAPE)

        augmented_video = aug.augment_video(frames)
        for nr, video in enumerate(augmented_video):
            video_features = np.zeros(
                shape=(1, MAX_FRAMES, NUM_FEATURES), dtype="float32"
            )
            video_mask = np.zeros(shape=(1, MAX_FRAMES,), dtype="bool")

            for i in range(min(MAX_FRAMES, video.shape[0])):
                frame = video[i]
                features = cnn.predict(frame[None, ...])
                video_features[:, i, :] = features
                video_mask[:, i] = 1
            labels[idx * NUM_AUG + nr,] = vocab_layer([row['tag']])
            all_features[idx * NUM_AUG + nr,] = video_features
            all_masks[idx * NUM_AUG + nr,] = video_mask

    return all_features, all_masks, labels


def learn(train_df, test_df):
    vocab_layer = tf.keras.layers.StringLookup(vocabulary=dp.vocab)

    cnn_model = get_cnn(preset='inception_v3')
    rnn_model = network.RecurrentNetwork(len(vocab_layer.get_vocabulary()))

    train_features, train_masks, train_labels = run_cnn(cnn_model, train_df, vocab_layer)
    test_features, test_masks, test_labels = run_cnn(cnn_model, test_df, vocab_layer)

    rnn_model.fit_model([train_features, train_masks], train_labels)

    loss, accuracy = rnn_model.evaluate([test_features, test_masks], test_labels)

    return rnn_model, cnn_model


sequence_model, conv = learn(train_data, test_data)
sequence_model.plot_results()
# analyse(hist)
