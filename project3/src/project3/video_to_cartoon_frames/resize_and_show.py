import cv2
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    """
    Resize the image to fit within a desired width and height while maintaining aspect ratio,
    and display the resized image.
    
    Args:
        image (np.ndarray): The input image to be resized and displayed.
    
    Returns:
        np.ndarray: The resized image.
    """
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img
