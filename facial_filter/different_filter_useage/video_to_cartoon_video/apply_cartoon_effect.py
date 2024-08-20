# project3/apply_cartoon_effect.py
import cv2
import mediapipe as mp

def apply_cartoon_effect(frame, stylizer):
    """
    Applies a cartoon effect to the provided frame using the given stylizer.

    Args:
        frame (numpy.ndarray): The frame to which the cartoon effect will be applied.
        stylizer (vision.FaceStylizer): An instance of the FaceStylizer class.

    Returns:
        numpy.ndarray: The stylized (cartoonized) frame.
    """
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        stylized_image = stylizer.stylize(mp_image)
        rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
        return rgb_stylized_image
    except Exception as e:
        print(f"Error applying cartoon effect: {e}")
        return frame