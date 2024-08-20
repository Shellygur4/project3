import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def apply_cartoon_effect(frame, stylizer):
    """
    Apply a cartoon effect to the given video frame using the MediaPipe Face Stylizer.
    
    Args:
        frame (np.ndarray): The input video frame to be stylized.
        stylizer (vision.FaceStylizer): The MediaPipe Face Stylizer instance.
    
    Returns:
        np.ndarray: The stylized video frame.
    """
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        stylized_image = stylizer.stylize(mp_image)
        rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
        return rgb_stylized_image
    except Exception as e:
        print(f"Error applying cartoon effect: {e}")
        return frame
