import pandas as pd
import tensorflow as tf
import helper_functions as hf
import augmentation as aug
import pathlib
import parameters
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

data_path = pathlib.Path.cwd() / 'data'
data_count = len(list(data_path.glob('*/*.mp4')))
vocab = [x.parts[-1] for x in data_path.iterdir() if x.is_dir()]


def create_dataset(data_path, labels):
    dataset = {}
    for label in labels:
        dataset[label] = [x.parts[-1] for x in data_path.glob('{}/*'.format(label))]

    for category, paths in dataset.items():
        summary = ", ".join(paths[:2])
        print("%-20s %4d videos (%s, ...)" % (category, len(paths), summary))
    return dataset


def split_dataset(dt, split):
    train_dataset = {}
    test_dataset = {}
    for label in dt:
        train_dataset[label] = [dt[label][i] for i in range(0, round(len(dt[label]) * split))]
        test_dataset[label] = [dt[label][i] for i in
                               range(round(len(dt[label]) * split), len(dt[label]))]
    return train_dataset, test_dataset


def split_dataframe(df):
    df = df.sample(frac=1)
    train, test = train_test_split(df, test_size=parameters.TEST_SIZE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def dataset_to_dataframe(dataset):
    data = {'video_name': [], 'tag': []}
    for label in dataset:
        for elem in dataset[label]:
            data['video_name'].append(elem)
            data['tag'].append(label)
    return pd.DataFrame(data)


def prepare_all_data(df, vocab):
    num_samples = len(df)
    video_paths = df["video_name"].values.tolist()
    vocab_layer = tf.keras.layers.StringLookup(vocabulary=vocab)
    labels = [vocab_layer(df["tag"].values)]

    for idx, path in enumerate(video_paths):
        original_frames = hf.load_video('data\\' + train_df['tag'][idx] + '\\' + path, max_frames=parameters.FRAMES,
                                        resize=parameters.VID_SHAPE)
        # aug1_frames = aug.center_crop(original_frames, parameters.vid_shape)
        # videos[parameters.aug_number * idx + 1,] = aug1_frames
    return original_frames, labels


# My dataset
dataset = create_dataset(data_path, vocab)

# train_data, test_data = split_dataset(dataset, train_test_split)
# train_df = dataset_to_dataframe(train_data)
# test_df = dataset_to_dataframe(test_data)
dataframe = dataset_to_dataframe(dataset)
train_df, test_df = split_dataframe(dataframe)

#
frames, labels = prepare_all_data(train_df, vocab)
aug_frames = aug.augment_video(frames)
# hf.to_gif(frames)
