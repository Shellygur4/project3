import os
import cv2
from main import main

def test_main():
    """
    Test the main function to ensure it processes the image correctly.
    """
    # Define paths used in the test
    input_image_path = "C:/Users/saeed/Desktop/project3-2/project3/src/project3/Child_Filter/data/input_image.jpg"
    output_directory = "C:/Users/saeed/Desktop/project3-2/project3/src/project3/Child_Filter/outputdata"

    # Clear the output directory before the test
    for file in os.listdir(output_directory):
        file_path = os.path.join(output_directory, file)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    # Run the main function
    main()

    # Check if an image was saved in the output directory
    output_files = os.listdir(output_directory)
    assert len(output_files) > 0, "No output file was created."

    # Verify that the output file is an image
    output_image_path = os.path.join(output_directory, output_files[0])
    output_image = cv2.imread(output_image_path)
    assert output_image is not None, "The saved output file is not a valid image."

