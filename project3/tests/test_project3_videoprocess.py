import os
import subprocess
import pytest
import numpy as np
from project3.videoprocess import resize_and_show, apply_cartoon_effect

def test_resize_and_show():
    # Create a dummy image
    image = np.zeros((100, 200, 3), dtype=np.uint8)  # Black image with width=200, height=100
    resized_image = resize_and_show(image)
    
    # Check the size of the resized image
    assert resized_image.shape[0] == 480 or resized_image.shape[1] == 480, "Image resize failed"
    
    # Check the resized image dimensions
    assert resized_image.shape[0] == 480 or resized_image.shape[1] == 480, "Image resize dimensions are incorrect"

def test_apply_cartoon_effect():
    # Mock the stylizer object
    class MockStylizer:
        def stylize(self, mp_image):
            return mp_image  # Just return the input image for testing
    
    # Create a dummy image
    frame = np.zeros((480, 480, 3), dtype=np.uint8)  # Black image with dimensions 480x480
    mock_stylizer = MockStylizer()
    cartoon_frame = apply_cartoon_effect(frame, mock_stylizer)
    
    # Check if the cartoon effect was applied (mocked here, so we just check if it's unchanged)
    assert np.array_equal(frame, cartoon_frame), "Cartoon effect application failed"


    print("Test successful")

if __name__ == "__main__":
    pytest.main()
