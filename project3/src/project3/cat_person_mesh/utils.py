# utils.py

import numpy as np
import cv2

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
