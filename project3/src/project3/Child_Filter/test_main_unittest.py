import unittest
import os
import cv2
from main import main

class TestChildlikeFilter(unittest.TestCase):

    def setUp(self):
        """
        Set up the test environment.
        """
        # Define paths used in the test
        self.input_image_path = "C:/Users/saeed/Desktop/project3-2/project3/src/project3/Child_Filter/data/input_image.jpg"
        self.output_directory = "C:/Users/saeed/Desktop/project3-2/project3/src/project3/Child_Filter/outputdata"
        
        # Clear the output directory before each test
        for file in os.listdir(self.output_directory):
            file_path = os.path.join(self.output_directory, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    def test_main(self):
        """
        Test the main function to ensure it processes the image correctly.
        """
        # Run the main function
        main()

        # Check if an image was saved in the output directory
        output_files = os.listdir(self.output_directory)
        self.assertTrue(len(output_files) > 0, "No output file was created.")

        # Verify that the output file is an image
        output_image_path = os.path.join(self.output_directory, output_files[0])
        output_image = cv2.imread(output_image_path)
        self.assertIsNotNone(output_image, "The saved output file is not a valid image.")

    def tearDown(self):
        """
        Clean up after tests if necessary.
        """
        # Additional cleanup can be added here if needed
        pass

if __name__ == "__main__":
    unittest.main()
