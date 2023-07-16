import os

import cv2
import numpy as np
from joblib import Parallel, delayed
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image

import utils

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    preprocessing_function=utils.image.add_noise_and_blur,
)


def augment_image(image_path, augmentation_times=3):
    # Load the image
    img = Image.open(image_path)
    img = np.array(img)
    img = utils.image.custom_zoom(img, (0.2, 0.2))

    # Generate augmented images and save to disk
    for i in range(augmentation_times):
        img_4d = img.reshape((1, ) + img.shape)
        for batch in datagen.flow(img_4d, batch_size=1):
            # Save augmented image to disk
            base_name = os.path.basename(image_path)
            dir_name = os.path.dirname(image_path)
            name, ext = os.path.splitext(base_name)
            augmented_image_path = os.path.join(dir_name,
                                                f"{name}_aug_{i}{ext}")

            cv2.imwrite(augmented_image_path, batch[0])
            # Break after the first image is created
            break


def augment_images(image_paths, augmentation_times=3):
    for image_path in image_paths:
        augment_image(image_path, augmentation_times)


def parallel_augmentation(image_paths, augmentation_times=3):
    # Use Joblib to parallelize the augmentation
    Parallel(n_jobs=4,
             verbose=1)(delayed(augment_image)(image_path, augmentation_times)
                        for image_path in image_paths)
