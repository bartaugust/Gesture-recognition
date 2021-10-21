import pandas as pd
import tensorflow as tf
import helper_functions as hf
import augmentation as aug
import pathlib
from parameters import *
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
    train, test = train_test_split(df, test_size=TEST_SIZE)
    return train.reset_index(drop=True), test.reset_index(drop=True)


def dataset_to_dataframe(dataset):
    data = {'video_name': [], 'tag': []}
    for label in dataset:
        for elem in dataset[label]:
            data['video_name'].append(elem)
            data['tag'].append(label)
    return pd.DataFrame(data)


def create_augmented_dataframe(df):
    videos = []
    labels = []
    for idx, row in df.iterrows():
        original_frames = hf.load_video('data\\' + row['tag'] + '\\' + row['video_name'''], max_frames=FRAMES,
                                        resize=VID_SHAPE)
        augmented_frames = aug.augment_video(original_frames)
        for video in augmented_frames:
            videos.append(video)
            labels.append(row['tag'])
    augmented_data = {'video': videos, 'tag': labels}
    augmented_dataframe = pd.DataFrame(augmented_data)
    augmented_dataframe = augmented_dataframe.sample(frac=1)
    return augmented_dataframe


# My dataset
def prepare_dataframes():
    dataset = create_dataset(data_path, vocab)
    dataframe = dataset_to_dataframe(dataset)
    return split_dataframe(dataframe)

