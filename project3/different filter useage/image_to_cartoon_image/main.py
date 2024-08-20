import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math

# Constants
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
    """
    Resize the image to maintain aspect ratio and display it.
    
    Args:
        image (np.ndarray): The input image to be resized and displayed.
    """
    h, w = image.shape[:2]
    if h < w:
        # Resize based on height
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        # Resize based on width
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    
    # Display the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)  # Wait for a key press
    cv2.destroyAllWindows()

# Define paths for input image (in data), model (face_stylizer), and output image (to ouput)
input_image_filename = r'project3\src\project3\image_to_cartoon_image\data\similing_woman.jpeg'
file_path = r'project3\src\project3\image_to_cartoon_image\face_stylizer.task'
output_folder = r'project3\src\project3\image_to_cartoon_image\output'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# Generate output file name based on input file name
input_file_base = os.path.splitext(os.path.basename(input_image_filename))[0]
output_image_filename = os.path.join(output_folder, f'{input_file_base}_cartoon.png')

# Load the input image
image = cv2.imread(input_image_filename)
if image is None:
    raise FileNotFoundError(f"Image file '{input_image_filename}' not found.")

# Face stylization setup
try:
    # Load the Face Stylizer model
    with open(file_path, 'rb') as f:
        model_data = f.read()
    
    # Set up model options
    base_options = python.BaseOptions(model_asset_buffer=model_data)
    options = vision.FaceStylizerOptions(base_options=base_options)
    
    # Create the Face Stylizer and apply the effect
    with vision.FaceStylizer.create_from_options(options) as stylizer:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        stylized_image = stylizer.stylize(mp_image)
        rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
        
        # Save the stylized image to the output file
        cv2.imwrite(output_image_filename, rgb_stylized_image)
        
        # Resize and display the stylized image
        resize_and_show(rgb_stylized_image)
    
    print("Face stylization successful and image saved.")
except Exception as e:
    print(f"Error during face stylization: {e}")
