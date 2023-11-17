import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from IPython.display import clear_output
from skimage.io import imread
from skimage.transform import resize
from PIL import Image, ImageOps
import cv2 as cv
import random

from CONSTS import IMG_H, IMG_W

images_path = 'C:\\Users\\szzmi\\Projekty\\DM-i-AI-2023\\tumor-segmentation\\data\\patients\\imgs'
labels_path = 'C:\\Users\\szzmi\\Projekty\\DM-i-AI-2023\\tumor-segmentation\\data\\patients\\labels'
unlabeled_path = 'C:\\Users\\szzmi\\Projekty\\DM-i-AI-2023\\tumor-segmentation\\data\\controls\\imgs'


SEED = 42


def get_labeled_files():
    control_images = []
    labels = []
    for file_name in os.listdir(images_path):
        control_images.append(images_path + "\\" + file_name)
    for file_name in os.listdir(labels_path):
        labels.append(labels_path + "\\" + file_name)
    control_images = sorted(control_images)
    labels = sorted(labels)
    return control_images, labels


def get_unlabeled_files():
    control_images = []
    for file_name in os.listdir(unlabeled_path):
        control_images.append(unlabeled_path + "\\" + file_name)
    control_images = sorted(control_images)
    return control_images


def read_images(files):
    return [imread(path, as_gray=True) for path in files]


def pad2square(img, black_white, mask):
    img_shape = img.shape
    r = img_shape[0]
    c = img_shape[1]
    img_as_img = Image.fromarray(img)
    mask_as_img = Image.fromarray(mask)

    # finding shoulders
    middle = len(img[0]) // 2
    K = 100
    crop = next((i for i,v in reversed(list(enumerate(black_white[:len(black_white)//3]))) if 1.  in [v[j] for j in range(middle - K//2, middle + (K//2) + 1)]), 0)
    cropped_img = img_as_img.crop((0, crop, c, r))
    cropped_mask = mask_as_img.crop((0, crop, c, r))

    return np.array(cropped_img), np.array(cropped_mask)


def crop(images, masks):
    resized_images = []
    resized_masks = []

    for im, mk in zip(images, masks):
        ret, thresh = cv.threshold(im, 0.95, 1, cv.THRESH_BINARY)

        cropped_im, cropped_mask = pad2square(im, thresh, mk)  # Make the image square
        resized_images.append(cropped_im)
        resized_masks.append(cropped_mask)
    return resized_images, resized_masks


def reshape(images, x, y):
    reshaped_images = []
    for img in images:
        img_as_img = Image.fromarray(img)
        img_shape = img.shape
        h = img_shape[0]
        w = img_shape[1]
        # left, top, right, bottom
        padding = ((x - w) // 2, 0, (x - w) // 2, (y - h))
        reshaped = ImageOps.expand(img_as_img, padding, fill=img[0, 0])
        reshaped_images.append(np.array(reshaped))
    return reshaped_images


def compress(images):
    reshaped_images = []
    for img in images:
        image = resize(img, output_shape=(IMG_H, IMG_W), mode='reflect', anti_aliasing=True)
        reshaped_images.append(image)

    return reshaped_images


def max_shape_from_image_list(images):
    return tuple(
        max([im.shape[i] for im in images]) for i in range(len(images[0].shape))
    )


def generate_overlay_2d(images, normalize=True):
    overlaid_shape = max_shape_from_image_list(images)
    overlaid_shape += (3,)
    overlaid = np.zeros(overlaid_shape, dtype=images[0].dtype)
    for im_i, im in enumerate(images):
        if normalize is True and np.abs(im.max()) > 0:
            overlaid[: im.shape[0], : im.shape[1], im_i] = im / im.max()
        else:
            overlaid[: im.shape[0], : im.shape[1], im_i] = im

    return overlaid


def display_overlay(resized_images, raw_images):

    amount = 15
    base_ids = random.sample(range(0, len(resized_images)), amount)
    overlay_ids = random.sample(range(0, len(resized_images)), amount)
    base_image = random.sample(resized_images, amount)
    overlay_image = random.sample(resized_images, amount)

    resize_base = random.choices(resized_images, k=amount)
    plt.figure(figsize=(50, 50))

    plt.imshow(generate_overlay_2d(resize_base, False), cmap='gray')

    for b, o in zip(base_ids, overlay_ids):
        base_raw = raw_images[b]
        overlay_raw = raw_images[o]
        img1 = resized_images[b]
        img2 = resized_images[o]
        plt.figure(figsize=(50, 50))
        subplots = 5
        plt.subplot(1, subplots, 1)
        # plt.title(img1)
        plt.imshow(base_raw, cmap='gray')

        plt.subplot(1, subplots, 2)
        # plt.title(img1)
        plt.imshow(overlay_raw, cmap='gray')

        plt.subplot(1, subplots, 3)
        # plt.title(img1)
        plt.imshow(img1, cmap='gray')
        plt.subplot(1, subplots, 4)
        # plt.title(img2)
        plt.imshow(img2, cmap='gray')

        # plt.subplot(1, subplots, 3)
        # plt.imshow(generate_overlay_2d([image, image2], False), cmap='gray')

        plt.subplot(1, subplots, 5)
        plt.imshow(generate_overlay_2d([img1, img2], False), cmap='gray')


def display_image(image):
    plt.figure(figsize=(15, 15))
    plt.imshow(image, cmap='gray')
    plt.show()


def display_many(images):
    i = 1
    plt.figure(figsize=(15, 15))
    for img in images:
        plt.subplot(1, len(images), i)
        plt.imshow(img, cmap='gray')
        i += 1
    plt.show()


def apply_threshold(masks):
    thresholded_masks = []

    for mk in masks:
        ret, thresh = cv.threshold(mk, 0.5, 1, cv.THRESH_BINARY)

        thresholded_masks.append(thresh)
    return thresholded_masks


def get_data():
    filenames_labeled, masks = get_labeled_files()
    filenames_unlabeled = get_unlabeled_files()

    images_labeled = read_images(filenames_labeled)[200:]
    masks_labeled = read_images(masks)

    images_unlabeled = read_images(filenames_unlabeled)
    masks_unlabeled = [np.zeros(img.shape) for img in images_unlabeled]

    cropped_images_labeled, cropped_masks_labeled = crop(images_labeled, masks_labeled)
    cropped_images_unlabeled, cropped_masks_unlabeled = crop(images_unlabeled, masks_unlabeled)

    compression_factor = 1.3
    w = 300
    h = 700

    reshaped_im_l = reshape(cropped_images_labeled, w, h)
    reshaped_m_l = reshape(cropped_masks_labeled, w, h)

    # display_many([reshaped_im_l[100], reshaped_m_l[100], reshaped_im_l[1], reshaped_m_l[1], reshaped_im_l[34], reshaped_m_l[34], reshaped_im_l[76], reshaped_m_l[76]])

    reshaped_im_ul = reshape(cropped_images_unlabeled, w, h)
    reshaped_m_ul = reshape(cropped_masks_unlabeled, w, h)

    resized_im_l = compress(reshaped_im_l)
    resized_m_l = compress(reshaped_m_l)

    resized_im_ul = compress(reshaped_im_ul)
    resized_m_ul = compress(reshaped_m_ul)

    # display_many([cropped_images_labeled[104], reshaped_im_l[104], reshaped_m_l[104], resized_im_l[104], resized_m_l[104]])
    # display_many([cropped_images_labeled[14], reshaped_im_l[14], reshaped_m_l[14], resized_im_l[14], resized_m_l[14]])

    # display_image(reshaped_im_ul[104])
    all_images = []
    all_images.extend(resized_im_l)
    all_images.extend(resized_im_ul)

    all_masks = []
    all_masks.extend(resized_m_l)
    all_masks.extend(resized_m_ul)

    all_masks_threshold = apply_threshold(all_masks)

    random.Random(SEED).shuffle(all_images)
    random.Random(SEED).shuffle(all_masks_threshold)

    # display_many([all_images[100], all_masks_threshold[100], all_images[1], all_masks_threshold[1], all_images[34], all_masks_threshold[34], all_images[76], all_masks_threshold[76]])

    return all_images, all_masks_threshold


if __name__ == '__main__':
    get_data()
