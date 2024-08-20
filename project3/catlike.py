
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

# Initialize drawing utilities
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def create_feature_masks(frame, face_landmarks):
    h, w, _ = frame.shape
    landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

    # Creating zeroed mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # Indices for specific facial features
    left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246]
    right_eye_indices = [362, 382, 381, 380, 385, 386, 387, 388, 263, 466]
    mouth_indices = [61, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13]
    nose_indices = [1, 2, 3, 4, 5, 6, 197, 195, 122, 96, 97, 98, 99, 240, 237, 220, 45]

    all_indices = left_eye_indices + right_eye_indices + mouth_indices + nose_indices

    # Create masks for the eyes, nose, and mouth
    for indices in [left_eye_indices, right_eye_indices, mouth_indices, nose_indices]:
        feature_points = np.array([landmarks[i] for i in indices])
        hull = cv2.convexHull(feature_points)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask

def apply_animal_effect(frame, face_landmarks, effect_texture):
    h, w, _ = frame.shape
    landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

    # Create a mask for the facial features (eyes, mouth, and nose)
    feature_mask = create_feature_masks(frame, face_landmarks)

    # Create a mask for the entire face
    mask = np.zeros((h, w), dtype=np.uint8)

    # Define indices for the face area, excluding key features
    face_indices = list(set(range(len(landmarks))) - set(np.where(feature_mask != 0)[0]))

    # Create convex hull for the face region
    face_points = np.array([landmarks[i] for i in face_indices if i < len(landmarks)])
    hull = cv2.convexHull(face_points)
    cv2.fillConvexPoly(mask, hull, 255)

    # Exclude the key features from the face mask
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(feature_mask))

    # Create a texture mask for blending
    effect_resized = cv2.resize(effect_texture, (w, h))
    effect_applied = cv2.bitwise_and(effect_resized, effect_resized, mask=mask)

    # Blend the effect with the original frame
    frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))
    frame = cv2.add(frame, effect_applied)

    return frame

def main(image_path, output_path, effect_texture_path):
    # Load the image and effect texture
    frame = cv2.imread(image_path)
    effect_texture = cv2.imread(effect_texture_path)

    if frame is None or effect_texture is None:
        print("Error loading images.")
        return

    # Convert image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to detect facial landmarks
    results = face_mesh.process(frame_rgb)
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        # Apply animal effect
        result_frame = apply_animal_effect(frame, face_landmarks, effect_texture)

        # Save the output image
        if cv2.imwrite(output_path, result_frame):
            print(f"Saved the output image to {output_path}")
        else:
            print(f"Error saving image to {output_path}")
    else:
        print("No face landmarks detected.")

if __name__ == "__main__":
    input_image_path = r"C:\Users\micha\Documents\PythonHW\project3-2\person.jpg"
    output_image_path = r"C:\Users\micha\Documents\PythonHW\project3-2\output_person.jpg"  # Ensure this includes filename and extension
    effect_texture_path = r"C:\Users\micha\Documents\PythonHW\project3-2\cat_texture.jpg"  # This should be a texture or pattern image

    main(input_image_path, output_image_path, effect_texture_path)