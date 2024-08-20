# project3/resize_and_show.py
import cv2
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    """
    Resizes the provided image to fit within the desired dimensions while maintaining aspect ratio.

    Args:
        image (numpy.ndarray): The image to be resized.

    Returns:
        numpy.ndarray: The resized image.
    """
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img