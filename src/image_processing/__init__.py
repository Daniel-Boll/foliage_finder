import multiprocessing as mp
import platform
from enum import Enum
from functools import partial

import cv2
from joblib import dump, load
from tqdm import tqdm

from utils import image

from . import (data_augmentation, extract_leaves, freeman_chain_code,
               local_binary_pattern, statistics)

# display = display.display_contours_from_features
extract_leaves = extract_leaves.extract_leaves
# extract_lbp_features = local_binary_pattern.extract_lbp_features

system = platform.system()


# Create a enum for step having (train, classify)
class Step(Enum):
    TRAIN = 1
    CLASSIFY = 2


def augment_images(image_paths, augmentation_times=3):
    data_augmentation.parallel_augmentation(image_paths, augmentation_times)
    # data_augmentation.augment_images(image_paths, augmentation_times)


def extract_features(image_paths, step=Step.TRAIN):
    # define the number of processes to spawn.
    # Here it is set to the number of cores.
    num_processes = mp.cpu_count()

    with mp.Pool(num_processes) as pool:
        # use partial to fix the values of the second and third arguments
        process_func = partial(extract_feature, step=step)
        features_list = list(
            tqdm(
                pool.imap(process_func, image_paths),
                total=len(image_paths),
                colour="green" if step == Step.TRAIN else "yellow",
                desc="Extracting features from images",
                unit="image",
            ))

    # Merge all dictionaries from features_list into one dictionary
    features = {
        k: v
        for feature_dict in features_list
        for k, v in feature_dict.items()
    }

    return features


def extract_feature(image_path, step=Step.TRAIN):
    features = {}

    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = image.custom_zoom(img, (0.2, 0.2))

    # Get the original contours and extract features
    contours = image.get_contours(image_path)
    features[image_path] = extract_features_from_contours(
        contours, image_path, step)

    return features


def extract_features_from_contours(contours, image_path, step):
    contour_features = []

    freeman_features = freeman_chain_code.extract_features(contours)
    statistical_features = statistics.extract_features(contours)
    local_binary_pattern_features = local_binary_pattern.extract_features(
        contours, image_path)

    for freeman_feature, statistical_feature in zip(freeman_features,
                                                    statistical_features):
        feature = {
            "chain_code": freeman_feature,
            "lbp": local_binary_pattern_features,
            **statistical_feature,
        }

        if step == Step.TRAIN:
            name = (image_path.split("\\")[1]
                    if system == "Windows" else image_path.split("/")[1])
            feature["class"] = name

        contour_features.append(feature)

    return contour_features


def save_features(features):
    dump(features, "features.pkl")


def load_features():
    return load("features.pkl")


def separate_leaves(image_paths, output_folder):
    for image_paths in tqdm(image_paths,
                            desc="Processing images",
                            unit="image"):
        extract_leaves(image_paths, output_folder)
