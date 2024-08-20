import cv2
import os
from datetime import datetime
from load_image import load_image
from save_image import save_image
from apply_childlike_filter import apply_childlike_filter

def main():
    """
    Main function to load an image, apply the childlike filter, and save the result.
    """
    # Path to the input image
    image_path = "C:/Users/saeed/Desktop/project3-2/project3/src/project3/Child_Filter/data/input_image.jpg"
    
    # Load the image
    image = load_image(image_path)
    if image is None:
        return

    # Apply the childlike filter
    result = apply_childlike_filter(image)

    # Generate a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"childlike_face_{timestamp}.jpg"

    # Path to the output directory
    output_directory = "C:/Users/saeed/Desktop/project3-2/project3/src/project3/Child_Filter/outputdata"
    
    # Ensure the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # Create the full output path
    output_path = os.path.join(output_directory, filename)
    
    # Save the processed image
    save_image(result, output_path)

    # Display the result
    cv2.imshow("Childlike Face", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
