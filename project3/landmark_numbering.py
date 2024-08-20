import cv2
import mediapipe as mp
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Paths for input images and output directory
path = "C:/Users/micha/Documents/PythonHW/project3-2/catperson images/"
path_out = "C:/Users/micha/Documents/PythonHW/project3-2/landmark_numbering/"

def detect_and_draw_face_mesh(image_path, output_path, scale_factor=1.0):
    """
    Detects facial landmarks in an image, numbers every sixth landmark, and saves the resized image.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the output image with numbered landmarks.
        scale_factor (float): Factor by which to resize the output image.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect facial landmarks
    results = face_mesh.process(image_rgb)

    # Draw every sixth landmark's index on the image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Draw every sixth landmark
                if idx % 15 == 0:
                    # Convert the landmark coordinates to pixel coordinates
                    h, w, _ = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    
                    # Draw the index number next to the landmark with smaller font
                    cv2.putText(image, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 255, 0), 1, cv2.LINE_AA)
                    
                    # Optionally, draw a small circle at each landmark (commented out)
                    # cv2.circle(image, (x, y), 2, (0, 255, 0), -1)

    # Resize the image to make it significantly bigger
    new_dim = (int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor))
    resized_image = cv2.resize(image, new_dim, interpolation=cv2.INTER_LINEAR)

    # Save the resized image with landmarks
    if not cv2.imwrite(output_path, resized_image):
        print(f"Error saving image with landmarks {output_path}")
    else:
        print(f"Saved the output image to {output_path}")

# Process images
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    output_file_path = os.path.join(path_out, filename)

    detect_and_draw_face_mesh(file_path, output_file_path, scale_factor=3.0)

# Release resources
face_mesh.close()
