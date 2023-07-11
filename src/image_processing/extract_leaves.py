import cv2
import numpy as np
import os
from tqdm import tqdm

def extract_leaves(image_path, output_folder):

    os.makedirs(output_folder, exist_ok=True)
    min_area_threshold = 100

    # Read the image with transparency information
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Split the image into channels
    channels = cv2.split(img)

    # Create a mask by thresholding the alpha channel
    _, mask = cv2.threshold(channels[3], 1, 255, cv2.THRESH_BINARY)

    # Find contours of the objects
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop through each contour and save the segmented object
    for i, contour in enumerate(contours):
        # Calculate the area of the contour
        area = cv2.contourArea(contour)

        # Skip small contours (likely noise or artifacts)
        if area < min_area_threshold:
            continue

        # Create a blank mask for the contour
        contour_mask = np.zeros_like(mask)

        # Draw the contour on the mask
        cv2.drawContours(contour_mask, [contour], 0, 255, -1)

        # Extract the object using the mask
        object_img = cv2.bitwise_and(img, img, mask=contour_mask)

        # Save the segmented object
        image_file = os.path.basename(image_path)
        object_name = f"{os.path.splitext(image_file)[0]}_{i}.png"
        output_path = os.path.join(output_folder, object_name)
        cv2.imwrite(output_path, object_img)
