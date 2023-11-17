import os

from CONSTS import IMG_H, IMG_W
from main import get_data, display_many

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
import cv2
from glob import glob
import scipy.io
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, CSVLogger
from unet import build_unet

""" Global parameters """
global COMPRESSION_FACTOR
global NUM_CLASSES
global CLASSES
global COLORMAP

""" Creating a directory """


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


""" Load and split the dataset """


def load_dataset(split=0.2):
    train_x, train_y = get_data()

    split_size = int(split * len(train_x))

    train_x, valid_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(train_y, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)


def read_image_mask(x, y):
    """ Image processing """
    # x = x / 255.0
    # x = x.astype(np.float32)
    x = np.expand_dims(x, axis=2)

    """ Mask processing """
    output = []
    for color in COLORMAP:
        cmap = np.equal(y, color)
        output.append(cmap)
    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)
    return x, output


def preprocess(x, y):
    def f(x, y):
        image, mask = read_image_mask(x, y)
        return image, mask

    image, mask = tf.numpy_function(f, [x, y], [tf.float32, tf.uint8])
    # image.set_shape([IMG_H, IMG_W, 1])
    mask.set_shape([IMG_H, IMG_W, NUM_CLASSES])

    return image, mask


def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(preprocess)
    # for x,y in dataset:
    #     preprocess(x,y)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(2)
    return dataset


def get_colormap():
    colormap = np.array(
        [
            0,
            1
        ])

    classes = [
        "no tumor",
        "tumor"
    ]

    return classes, colormap


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    NUM_CLASSES = 2
    COMPRESSION_FACTOR = 1.3
    input_shape = (IMG_H, IMG_W, 1)

    batch_size = 12
    lr = 1e-4
    num_epochs = 10

    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "data.csv")

    """ Loading the dataset """
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset()
    print(
        f"Train: {len(train_x)}/{len(train_y)} - Valid: {len(valid_x)}/{len(valid_y)} - Test: {len(test_x)}/{len(test_x)}")
    print("")

    """ Process the colormap """
    CLASSES, COLORMAP = get_colormap()

    """ Dataset Pipeline """
    train_dataset = tf_dataset(train_x, train_y, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=batch_size)

    # for a in train_dataset.take(3):
    #
    #     display_many([a])


    """ Model """
    model = build_unet(input_shape, NUM_CLASSES)
    # model.load_weights(model_path)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr)
    )
    model.summary()

    """ Training """
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path, append=True),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
    ]

    model.fit(train_dataset,
              validation_data=valid_dataset,
              epochs=num_epochs,
              callbacks=callbacks
              )
