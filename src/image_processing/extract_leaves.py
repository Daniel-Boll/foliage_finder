import cv2
import os
import numpy as np

def extract_leaves(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".bmp"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply thresholding to create a binary image
            _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Find contours in the binary image
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Create a mask for the leaves
            mask = cv2.drawContours(np.zeros(image.shape[:2], dtype=np.uint8), contours, -1, (255, 255, 255), thickness=cv2.FILLED)

            # Convert the image and mask to RGBA
            image_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            mask_rgba = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA)

            # Apply the mask to the RGBA image
            result = cv2.bitwise_and(image_rgba, mask_rgba)

            output_path = os.path.join("output_segmented_images", filename.rsplit(".", 1)[0] + ".png")
            cv2.imwrite(output_path, result)

extract_leaves(input_folder, "output_segmented_images")
