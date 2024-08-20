import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize the MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,             # Process images in static mode (not streaming)
    max_num_faces=1,                    # Limit to 1 face per image
    refine_landmarks=True,              # Refine landmark positions for better accuracy
    min_detection_confidence=0.5        # Minimum confidence threshold for face detection
)

# Paths for input images and output directories
path = "C:/Users/micha/Documents/PythonHW/project3-1/catperson images/"
path_out_mesh = "C:/Users/micha/Documents/PythonHW/project3-2/OutputFaceMesh/"
path_out_smile = "C:/Users/micha/Documents/PythonHW/project3-2/OutputSmile/"

def get_lip_color(frame, lip_indices, landmarks):
    """
    Extract the average lip color from the specified lip indices.

    Args:
        frame (numpy.ndarray): The image frame containing the face.
        lip_indices (list): List of landmark indices defining the lip region.
        landmarks (list): List of (x, y) landmark coordinates.

    Returns:
        tuple: The average BGR color of the lip region.
    """
    lip_points = np.array([landmarks[i] for i in lip_indices])
    x_min, y_min = np.min(lip_points, axis=0)
    x_max, y_max = np.max(lip_points, axis=0)
    lip_region = frame[y_min:y_max, x_min:x_max]
    avg_color = cv2.mean(lip_region)[:3]
    return avg_color

# First pass: Save images with face mesh
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    frame = cv2.imread(file_path)

    if frame is None:
        print(f"Error loading image {file_path}")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        print(f"No face landmarks detected in {filename}")
        continue

    face_landmarks = results.multi_face_landmarks[0]
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))
    connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, color=(255, 0, 0))

    # Draw the face mesh on the frame
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp_face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )

    # Save the image with face mesh
    new_name_mesh = os.path.join(path_out_mesh, filename)
    if not cv2.imwrite(new_name_mesh, frame):
        print(f"Error saving image with mesh {new_name_mesh}")

# Second pass: Save images with smile modification
for filename in os.listdir(path):
    file_path = os.path.join(path, filename)
    frame = cv2.imread(file_path)

    if frame is None:
        print(f"Error loading image {file_path}")
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if not results.multi_face_landmarks:
        print(f"No face landmarks detected in {filename}")
        continue

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = frame.shape
    landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

    # Define lip indices for outer and inner lips
    outer_lip_indices = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
    inner_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 324, 318]

    # Define the smile factor for modifying the lip shape
    smile_factor = 0.5
    for i in outer_lip_indices + inner_lip_indices:
        landmarks[i] = (
            landmarks[i][0] + int((landmarks[i][0] - landmarks[0][0]) * smile_factor),
            landmarks[i][1] - int(abs(landmarks[i][0] - landmarks[0][0]) * smile_factor * 0.2)
        )

    # Create convex hulls for outer and inner lips
    outer_lip_hull = cv2.convexHull(np.array([landmarks[i] for i in outer_lip_indices]))
    inner_lip_hull = cv2.convexHull(np.array([landmarks[i] for i in inner_lip_indices]))

    # Extract lip color and apply to the modified smile region
    lip_color = get_lip_color(frame, outer_lip_indices + inner_lip_indices, landmarks)
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.drawContours(mask, [outer_lip_hull], -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.drawContours(mask, [inner_lip_hull], -1, (0, 0, 0), thickness=cv2.FILLED)

    mask_bool = mask[:, :, 0] == 255
    frame[mask_bool] = lip_color

    # Draw the contours with the same color as the lips
    cv2.drawContours(frame, [outer_lip_hull], -1, tuple(map(int, lip_color)), 2)
    cv2.drawContours(frame, [inner_lip_hull], -1, tuple(map(int, lip_color)), 2)

    # Save the image with smile modification
    new_name_smile = os.path.join(path_out_smile, filename)
    if not cv2.imwrite(new_name_smile, frame):
        print(f"Error saving image with smile {new_name_smile}")

# Release resources
face_mesh.close()
