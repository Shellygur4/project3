import cv2

def load_image(image_path):
    """
    Loads an image from the specified path.

    Args:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image, or None if the image cannot be loaded.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
    return image
