import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow import keras
import data_preparation as dp
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


vocab_layer = tf.keras.layers.StringLookup(vocabulary=dp.vocab)


def run_cnn(cnn, df):
    num_samples = len(df)
    all_features = np.zeros(
        shape=(num_samples, MAX_FRAMES, NUM_FEATURES), dtype="float32"
    )
    all_masks = np.zeros(shape=(num_samples, MAX_FRAMES), dtype="bool")

    labels = np.zeros(shape=(num_samples, 1))

    for idx, row in df.iterrows():
        frames = hf.load_video('data\\' + row['tag'] + '\\' + row['video_name'], max_frames=FRAMES,
                               resize=VID_SHAPE)
        video_features = np.zeros(
            shape=(1, MAX_FRAMES, NUM_FEATURES), dtype="float32"
        )
        video_mask = np.zeros(shape=(1, MAX_FRAMES), dtype="bool")

        for i in range(min(MAX_FRAMES, frames.shape[0])):
            frame = frames[i]
            features = cnn.predict(frame[None, ...])
            video_features[:, i, :] = features
            video_mask[:, i] = 1
        labels[idx, ] = vocab_layer([row['tag']])
        all_features[idx, ] = video_features
        all_masks[idx, ] = video_mask

    return all_features, all_masks, labels


def learn():
    pass


train_df, test_df = dp.prepare_dataframes()

cnn_model = get_cnn()
rnn_model = get_rnn()

train_features, train_masks, train_labels = run_cnn(cnn_model, train_df)
test_features, test_masks, test_labels = run_cnn(cnn_model, test_df)


def get_sequence_model():
    class_vocab = vocab_layer.get_vocabulary()

    frame_features_input = keras.Input((MAX_FRAMES, NUM_FEATURES))
    mask_input = keras.Input((MAX_FRAMES,), dtype="bool")

    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    x = keras.layers.GRU(16, return_sequences=True)(
        frame_features_input, mask=mask_input
    )
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnn_model


def run_experiment():
    # filepath = "/tmp"
    # checkpoint = keras.callbacks.ModelCheckpoint(
    #     filepath, save_weights_only=True, save_best_only=True, verbose=1
    # )

    seq_model = get_sequence_model()
    history = seq_model.fit(
        [train_features, train_masks],
        train_labels,
        validation_split=0.3,
        epochs=EPOCHS,
        # callbacks=[checkpoint],
    )

    # seq_model.load_weights(filepath)
    _, accuracy = seq_model.evaluate([test_features, test_masks], test_labels)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    return history, seq_model


_, sequence_model = run_experiment()
