# project3/load_face_stylizer_model.py
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def load_face_stylizer_model(model_path):
    """
    Loads the MediaPipe Face Stylizer model from the specified path.

    Args:
        model_path (str): Path to the Face Stylizer model file.

    Returns:
        vision.FaceStylizer: An instance of the FaceStylizer class if successful, else None.
    """
    try:
        with open(model_path, 'rb') as f:
            model_data = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceStylizerOptions(base_options=base_options)
        stylizer = vision.FaceStylizer.create_from_options(options)
        return stylizer
    except Exception as e:
        print(f"Error loading Face Stylizer model: {e}")
        return None