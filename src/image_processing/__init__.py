import platform
from enum import Enum

import cv2
import numpy as np
from tqdm import tqdm

from utils import image

from . import (data_augmentation, display, extract_leaves, freeman_chain_code,
               statistics)

display = display.display_contours_from_features
extract_leaves = extract_leaves.extract_leaves

system = platform.system()


# Create a enum for step having (train, classify)
class Step(Enum):
    TRAIN = 1
    CLASSIFY = 2


def extract_features(image_paths, step=Step.TRAIN, augmentation_times=3):
    features = {}

    for image_path in tqdm(image_paths, desc="Extracting features"):
        # Load the image
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img = image.custom_zoom(img, (0.2, 0.2))

        # Get the original contours and extract features
        contours = image.get_contours(image_path)
        features[image_path] = extract_features_from_contours(
            contours, image_path, step)

        # Generate augmented images and extract features
        for _ in range(augmentation_times):
            # ImageDataGenerator requires 4D input, so we add a new dimension
            # Also, it expects the format [samples][width][height][channels]
            img_4d = np.expand_dims(img, axis=0)
            for batch in data_augmentation.datagen.flow(img_4d, batch_size=1):
                # Here batch[0] is the augmented image
                augmented_img = (batch[0] * 255).astype(
                    np.uint8)  # Convert back to [0, 255] range

                # Calculate the contours for the augmented image
                # and extract features
                contours = image.get_contours(augmented_img)
                augmented_features = extract_features_from_contours(
                    contours, image_path, step)
                features[f"{image_path}_aug"] = augmented_features

                # We are using flow to create one augmented image at a time
                # so we need to break the loop after the first image is created
                break
    return features


def extract_features_from_contours(contours, image_path, step):
    contour_features = []

    freeman_features = freeman_chain_code.extract_features(contours)
    statistical_features = statistics.extract_features(contours)

    for freeman_feature, statistical_feature in zip(freeman_features,
                                                    statistical_features):
        feature = {
            "chain_code": freeman_feature,
            **statistical_feature,
        }

        if step == Step.TRAIN:
            name = (image_path.split("\\")[1]
                    if system == "Windows" else image_path.split("/")[1])
            feature["class"] = name

        contour_features.append(feature)

    return contour_features


def separate_leaves(image_paths, output_folder):
    for image_paths in tqdm(image_paths,
                            desc="Processing images",
                            unit="image"):
        extract_leaves(image_paths, output_folder)
