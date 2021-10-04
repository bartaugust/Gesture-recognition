import tensorflow as tf
from tensorflow_docs.vis import embed
import tensorflow_hub as hub
import pathlib
import cv2
import numpy as np
import imageio
from IPython import display

#Parameters
vid_shape = (500, 500)
frames = 100

data_path = pathlib.Path.cwd() / 'data'
data_count = len(list(data_path.glob('*/*.mp4')))
labels = [x.parts[-1] for x in data_path.iterdir() if x.is_dir()]


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=frames, resize=vid_shape):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, fps=30)
    return embed.embed_file('./animation.gif')


def create_dataset():
    dataset = {}
    for label in labels:
        dataset[label] = [x.parts[-1] for x in data_path.glob('{}/*'.format(label))]

    for category, paths in dataset.items():
        summary = ", ".join(paths[:2])
        print("%-20s %4d videos (%s, ...)" % (category, len(paths), summary))
    return dataset


# My dataset
dataset = create_dataset()

# Sample transformed video
czesc = list(data_path.glob('cześć/*'))
sample_video = load_video(str(czesc[0]))
to_gif(sample_video)
