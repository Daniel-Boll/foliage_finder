import cv2
from tqdm import tqdm

from . import display, freeman_chain_code, statistics, extract_leaves

display = display.display_contours_from_features
extract_leaves = extract_leaves.extract_leaves

def extract_features(image_paths):
    features = {}

    for image_path in tqdm(image_paths, desc="Extracting features"):
        # Load the image
        image = cv2.imread(image_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(image)

        # Calculate contours
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        contour_features = []

        freeman_features = freeman_chain_code.extract_features(contours)
        statistical_features = statistics.extract_features(contours)

        for freeman_feature, statistical_feature in zip(
                freeman_features, statistical_features):
            feature = {
                "chain_code": freeman_feature,
                **statistical_feature,
            }
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
    for image_paths in tqdm(image_paths, desc="Processing images", unit="image"):
        extract_leaves(image_paths, output_folder)