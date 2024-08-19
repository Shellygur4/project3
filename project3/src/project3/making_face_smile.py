import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# Initialize drawing utilities for landmarks
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def detect_and_smile_face(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Process the image to detect facial landmarks
    results = face_mesh.process(image_rgb)
    
    # Create an output image
    output_image = image.copy()

    # Draw and modify the facial landmarks to simulate a smile
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw the original landmarks
            mp_drawing.draw_landmarks(
                image=output_image, 
                landmark_list=face_landmarks, 
                connections=mp_face_mesh.FACEMESH_TESSELATION, 
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            # Convert landmarks to numpy array for easy manipulation
            landmarks = np.array([(landmark.x * w, landmark.y * h) for landmark in face_landmarks.landmark])
            
            # Get the indices of mouth landmarks (these may need to be adjusted)
            mouth_indices = list(range(61, 81))  # Approximate indices for mouth landmarks

            # Modify mouth landmarks to create a smiling effect
            for idx in mouth_indices:
                if idx < len(landmarks):
                    landmarks[idx][1] -= 5  # Move mouth landmarks up to simulate a smile

            # Draw the modified mouth landmarks
            for i in range(len(mouth_indices) - 1):
                cv2.line(output_image, tuple(map(int, landmarks[mouth_indices[i]])), tuple(map(int, landmarks[mouth_indices[i+1]])), (0, 255, 0), 2)
            # Add lines to connect the mouth landmarks in a smiling curve

    # Save the image with the smile effect
    cv2.imwrite(output_path, output_image)
    print(f"Saved the output image to {output_path}")

if __name__ == "__main__":
    # Example usage: simulate a smile on an image and save the result
    input_image_path = r"src\face_image.jpeg"  # Update with your input image path
    output_image_path = r"src\smiling_face_image.jpeg"  # Update with your output image path
    detect_and_smile_face(input_image_path, output_image_path)
