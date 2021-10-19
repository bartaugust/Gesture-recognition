from vidaug import augmentors as va
import numpy as np


def center_crop(frames, shape):
    seq = va.Sequential([
        va.CenterCrop(size=shape),
    ])
    augmented_video = seq(frames)
    return augmented_video


def augmentation_1(frames):
    seq = va.Sequential([
        va.RandomRotate(degrees=10),

    ])

    augmented_video = seq(frames)
    return augmented_video


def augmentation_2(frames):
    seq = va.Sequential([
        va.HorizontalFlip()  # horizontally flip the video with 50% probability
    ])

    augmented_video = seq(frames)
    return augmented_video


def augmentation_3(frames, shape):
    sometimes = lambda aug: va.Sometimes(0.5, aug)  # Used to apply augmentor with 50% probability
    seq = va.Sequential([
        va.RandomCrop(size=shape),  # randomly crop video with a size of (240 x 180)
        va.RandomRotate(degrees=10),  # randomly rotates the video with a degree randomly choosen from [-10, 10]
        sometimes(va.HorizontalFlip())  # horizontally flip the video with 50% probability
    ])

    augmented_video = seq(frames)
    return augmented_video


def augment_video(frames):
    all_frames = [frames]
    all_frames.append(augmentation_1(frames))
    all_frames.append(augmentation_2(frames))
    return np.array(all_frames)
