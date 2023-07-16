import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def compute_local_binary_pattern(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, 8, 1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(),
                               bins=np.arange(0, 59),
                               range=(0, 58))

    return lbp_hist


def extract_features(contours, image_path):
    lbp_features = []
    image = cv2.imread(image_path)

    for contour in contours:
        if len(contour) < 5:  # Skip contours with fewer than 5 points
            continue
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], 0, 255, thickness=cv2.FILLED)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        # Append a list to the current contour
        lbp_features.append(list(compute_local_binary_pattern(masked_image)))

    return lbp_features
