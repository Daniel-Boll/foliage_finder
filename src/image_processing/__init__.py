from enum import Enum

import cv2
import numpy as np
from tqdm import tqdm

from . import display, extract_leaves, freeman_chain_code, statistics

display = display.display_contours_from_features
extract_leaves = extract_leaves.extract_leaves


def image_has_alpha_channel(image):
    """
    Check if an image has an alpha channel.

    Parameters:
    image (numpy.ndarray): The image array.

    Returns:
    bool: True if the image has an alpha channel, False otherwise.
    """
    return image.shape[-1] == 4


def get_contours(image, min_contour_area=3000):
    """
    Extract contours from an image.
    Uses morphological transformations and Canny edge detection to improve
    contour finding.
    Uses metadata from the image to determine whether or not to split the
    alpha channel.

    Parameters:
    image (str): The image file path.
    min_contour_area (int): The minimum area for a contour to be considered
    valid.
                            Default is 1000.
    """
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)

    if image_has_alpha_channel(img):
        # Split the image into channels
        channels = cv2.split(img)

        # Create a mask by thresholding the alpha channel
        _, mask = cv2.threshold(channels[3], 1, 255, cv2.THRESH_BINARY)

        # # Morphological transformations to get rid of noise
        # kernel = np.ones((5, 5), np.uint8)
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        #
        # # Canny Edge Detection
        # edges = cv2.Canny(opening, 100, 200)

        # Find contours of the objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter out smaller contours based on area
        contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
        ]

        return contours
    else:
        # Preprocess the image
        preprocessed_image = preprocess_image(img)

        # Morphological transformations to get rid of noise
        # kernel = np.ones((5, 5), np.uint8)
        # opening = cv2.morphologyEx(preprocessed_image, cv2.MORPH_OPEN, kernel)
        #
        # # Canny Edge Detection
        # edges = cv2.Canny(opening, 100, 200)

        # Calculate contours
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter out smaller contours based on area
        contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
        ]

        return contours


# Create a enum for step having (train, classify)
class Step(Enum):
    TRAIN = 1
    CLASSIFY = 2


def extract_features(image_paths, step=Step.TRAIN):
    features = {}

    for image_path in tqdm(image_paths, desc="Extracting features"):
        # Load the image
        name = None if step == Step.CLASSIFY else image_path.split("/")[1]

        contours = get_contours(image_path)

        contour_features = []

        freeman_features = freeman_chain_code.extract_features(contours)
        statistical_features = statistics.extract_features(contours)

        for freeman_feature, statistical_feature in zip(
                freeman_features, statistical_features):
            feature = {
                "chain_code": freeman_feature,
                **statistical_feature,
            }

            if step == Step.TRAIN:
                feature["class"] = name

            contour_features.append(feature)

        features[image_path] = contour_features

    return features


def preprocess_image(image):
    """
    Preprocess the image before extracting contours.

    Parameters:
    image (numpy.ndarray): The image array.

    Returns:
    numpy.ndarray: The preprocessed image.
    """
    # Convert the image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Optionally smooth the image
    blurred = cv2.GaussianBlur(grayscale, (7, 7), 0)

    # Apply Otsu's thresholding
    _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

    return thresholded


def separete_leaves(image_paths, output_folder):
    for image_paths in tqdm(image_paths,
                            desc="Processing images",
                            unit="image"):
        extract_leaves(image_paths, output_folder)
