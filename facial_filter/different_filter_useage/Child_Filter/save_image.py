import cv2

def save_image(image, output_path):
    """
    Saves the processed image to the specified path.

    Args:
        image (numpy.ndarray): The processed image.
        output_path (str): The path where the image will be saved.
    
    Returns:
        bool: True if the image was saved successfully, False otherwise.
    """
    try:
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Processed image saved to {output_path}")
        else:
            print(f"Failed to save the image to {output_path}")
        return success
    except Exception as e:
        print(f"An error occurred while saving the image: {str(e)}")
        return False
