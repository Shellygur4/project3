import cv2
import numpy as np
import mediapipe as mp

def apply_childlike_filter(image):
    """
    Applies a childlike filter to the input image.

    This function enlarges the eyes, smooths the skin, and adds a subtle blush effect
    to the cheeks to give the face a childlike appearance.

    Args:
        image (numpy.ndarray): The input image in BGR format.

    Returns:
        numpy.ndarray: The processed image with the childlike filter applied.
    """
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        print("No face detected")
        return image

    h, w = image.shape[:2]
    landmarks = results.multi_face_landmarks[0].landmark

    def get_landmark(index):
        """
        Returns the (x, y) coordinates of a facial landmark.

        Args:
            index (int): The index of the facial landmark.

        Returns:
            tuple: (x, y) coordinates of the landmark.
        """        
        return (int(landmarks[index].x * w), int(landmarks[index].y * h))

    # Enlarge eyes
    left_eye_indices = [33, 133, 157, 158, 159, 160, 161, 173, 246]
    right_eye_indices = [362, 263, 386, 387, 388, 389, 390, 398, 466]

    for eye_indices in [left_eye_indices, right_eye_indices]:
        eye_center = np.mean([get_landmark(i) for i in eye_indices], axis=0).astype(int)
        for idx in eye_indices:
            pt = get_landmark(idx)
            vec = pt - eye_center
            pt_new = (eye_center + vec * 1.2).astype(int)
            cv2.line(image, tuple(pt), tuple(pt_new), (0, 0, 0), 1)
            cv2.circle(image, tuple(pt_new), 1, (0, 0, 0), -1)

    # Apply skin smoothing
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    mask = np.zeros(image.shape[:2], np.uint8)
    
    # Use a predefined list of face oval indices
    face_oval_indices = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
    
    face_oval_points = np.array([get_landmark(i) for i in face_oval_indices], dtype=np.int32)
    cv2.fillConvexPoly(mask, face_oval_points, 255)
    
    # Calculate the center of the face region
    face_center_x = np.mean([point[0] for point in face_oval_points]).astype(int)
    face_center_y = np.mean([point[1] for point in face_oval_points]).astype(int)
    face_center = (face_center_x, face_center_y)
    
    # Apply seamless cloning with the calculated face center
    image = cv2.seamlessClone(blur, image, mask, face_center, cv2.NORMAL_CLONE)

    return image
