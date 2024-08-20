# image_processing.py

import cv2
import numpy as np
import mediapipe as mp
from utils import get_lip_color

def draw_face_mesh(frame, face_landmarks):
    """
    Draw face mesh on the image frame.

    Args:
        frame (numpy.ndarray): The image frame.
        face_landmarks: Face landmarks from MediaPipe.
    """
    drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, circle_radius=2, color=(0, 255, 0))
    connection_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=2, color=(255, 0, 0))

    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=drawing_spec,
        connection_drawing_spec=connection_spec
    )

def modify_smile(frame, face_landmarks, smile_factor, outer_lip_indices, inner_lip_indices):
    """
    Modify the smile on the image frame.

    Args:
        frame (numpy.ndarray): The image frame.
        face_landmarks: Face landmarks from MediaPipe.
        smile_factor (float): Factor by which to modify the smile.
        outer_lip_indices (list): Indices for the outer lip landmarks.
        inner_lip_indices (list): Indices for the inner lip landmarks.
    """
    h, w, _ = frame.shape
    landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks.landmark]

    for i in outer_lip_indices + inner_lip_indices:
        landmarks[i] = (
            landmarks[i][0] + int((landmarks[i][0] - landmarks[0][0]) * smile_factor),
            landmarks[i][1] - int(abs(landmarks[i][0] - landmarks[0][0]) * smile_factor * 0.2)
        )

    outer_lip_hull = cv2.convexHull(np.array([landmarks[i] for i in outer_lip_indices]))
    inner_lip_hull = cv2.convexHull(np.array([landmarks[i] for i in inner_lip_indices]))

    lip_color = get_lip_color(frame, outer_lip_indices + inner_lip_indices, landmarks)
    mask = np.zeros(frame.shape, dtype=np.uint8)
    cv2.drawContours(mask, [outer_lip_hull], -1, (255, 255, 255), thickness=cv2.FILLED)
    cv2.drawContours(mask, [inner_lip_hull], -1, (0, 0, 0), thickness=cv2.FILLED)

    mask_bool = mask[:, :, 0] == 255
    frame[mask_bool] = lip_color

    cv2.drawContours(frame, [outer_lip_hull], -1, tuple(map(int, lip_color)), 2)
    cv2.drawContours(frame, [inner_lip_hull], -1, tuple(map(int, lip_color)), 2)
