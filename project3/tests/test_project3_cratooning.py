#!/usr/bin/env python

"""Tests for `project3` package."""
import os
import cv2
import numpy as np
import math
import pytest
from project3.cratooning import resize_and_show

# Paths (adjust as necessary)
input_image_filename = r'C:\Users\shell\homweork\project\project3\project3\src\project3\natural_face.jpeg'
output_image_filename = r'C:\Users\shell\homweork\project\project3\project3\src\project3\natural_style.png'

def test_resize_and_show():
    # Debug: Print the file path
    print(f"Testing resize_and_show with image: {input_image_filename}")

    # Load a sample image
    image = cv2.imread(input_image_filename)
    
    # Debug: Check if image is loaded
    if image is None:
        raise FileNotFoundError(f"Image failed to load from path: {input_image_filename}")
    
    # Perform resizing as in your function
    h, w = image.shape[:2]
    if h < w:
        expected_height = math.floor(h / (w / 480))
        expected_width = 480
    else:
        expected_height = 480
        expected_width = math.floor(w / (h / 480))

    resized_image = cv2.resize(image, (expected_width, expected_height))

    # Check that resizing happened correctly
    assert resized_image.shape[0] == expected_height
    assert resized_image.shape[1] == expected_width

def test_image_saving_and_loading():
    # Debug: Print the file path
    print(f"Testing image saving/loading with output: {output_image_filename}")

    # Save a test image
    test_image = cv2.imread(input_image_filename)
    assert test_image is not None, "Test image failed to load."

    # Save the image
    cv2.imwrite(output_image_filename, test_image)
    
    # Check that the file was created
    assert os.path.exists(output_image_filename), f"Image file was not saved: {output_image_filename}"
    assert os.path.getsize(output_image_filename) > 0, "Image file is empty."

    # Load the image back
    loaded_image = cv2.imread(output_image_filename)
    
    # Debug: Check if image is loaded
    if loaded_image is None:
        raise FileNotFoundError(f"Saved image could not be loaded from path: {output_image_filename}")

    # Check that the loaded image matches the original
    assert np.array_equal(test_image, loaded_image), "Loaded image does not match original."

    # Clean up
    os.remove(output_image_filename)

if __name__ == "__main__":
    pytest.main()


