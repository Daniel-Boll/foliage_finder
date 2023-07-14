import cv2
import numpy as np
from PIL import Image


def add_noise_and_blur(img):
    # # Convert image to float and scale to [0, 1]
    # img = img.astype(float) / 255.0
    #
    # # Add salt-and-pepper noise to the image
    # img = random_noise(
    #     img, mode="s&p",
    #     amount=0.1)  # you can change the amount value as per your requirement

    # Since the output of random_noise is float data in range [0, 1],
    # we need to convert it back to uint8 with range [0, 255]
    img = np.array(255 * img, dtype="uint8")

    # Add a random Gaussian blur of random size with random kernel sizes
    # and random sigma
    ksize = np.random.choice([1, 3, 5, 7, 9])
    img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    return img


def custom_zoom(img, zoom_range):
    # Convert the image to grayscale to create a mask for the leaf
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Identify the bounding box of the leaf
    rows, cols = np.where(img_gray < 255)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # Compute the zoom factor and ensure it doesn't zoom into the white background
    zoom_factor = np.random.uniform(*zoom_range)
    zoom_factor = max(
        min(zoom_factor, min_row / img.shape[0], min_col / img.shape[1]), 1)

    # Compute the size of the new image
    new_size = (
        int(img.shape[1] * zoom_factor),
        int(img.shape[0] * zoom_factor),
    )

    # Resize the image
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)

    # If the image is smaller than the original size, pad it with white pixels
    if img.shape[0] < img.shape[1]:
        padding = max((img.shape[1] - img.shape[0]) // 2, 0)
    else:
        padding = max((img.shape[0] - img.shape[1]) // 2, 0)
    img = cv2.copyMakeBorder(
        img,
        padding,
        padding,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=[255, 255, 255],  # white color for 3-channel image
    )

    # Convert the numpy array back to a PIL Image
    img_pil = Image.fromarray(img)

    return img_pil


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
       image (str or numpy.ndarray): The image file path or an open image.
       min_contour_area (int): The minimum area for a contour to be considered
       valid.
                               Default is 1000.
    """
    if isinstance(image, str):
        # If the image is a file path, read the image
        img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    elif isinstance(image, np.ndarray):
        # If the image is already an open image, just assign it to img
        img = image
    else:
        # Multi line this error
        raise TypeError("""
            The image argument must be either a file path (str)
            or an open image (numpy.ndarray).
            """)

    if image_has_alpha_channel(img):
        # Split the image into channels
        channels = cv2.split(img)

        # Create a mask by thresholding the alpha channel
        _, mask = cv2.threshold(channels[3], 1, 255, cv2.THRESH_BINARY)

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

        # Calculate contours
        contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter out smaller contours based on area
        contours = [
            cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area
        ]

        return contours


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
