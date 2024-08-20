# face_mesh_setup.py

import mediapipe as mp

def initialize_face_mesh():
    """
    Initialize the MediaPipe Face Mesh solution with specific parameters.
    
    Returns:
        FaceMesh: Initialized MediaPipe FaceMesh object.
    """
    mp_face_mesh = mp.solutions.face_mesh
    return mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    )
