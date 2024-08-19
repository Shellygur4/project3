import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Constants
DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
VIDEO_INPUT_PATH = r'C:\Users\shell\homweork\project\project3\project3\src\project3\michal2.mp4'
FACE_STYLIZER_MODEL_PATH = r'C:\Users\shell\homweork\project\project3\project3\face_stylizer.task'
NUM_FRAMES = 3  # Number of frames to capture
TIME_INTERVALS = [0, 1.5, 3]  # Time intervals (in seconds) to capture frames

def resize_and_show(image):
    h, w = image.shape[:2]
    if h < w:
        img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h / (w / DESIRED_WIDTH))))
    else:
        img = cv2.resize(image, (math.floor(w / (h / DESIRED_HEIGHT)), DESIRED_HEIGHT))
    return img

def apply_cartoon_effect(frame, stylizer):
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        stylized_image = stylizer.stylize(mp_image)
        rgb_stylized_image = cv2.cvtColor(stylized_image.numpy_view(), cv2.COLOR_BGR2RGB)
        return rgb_stylized_image
    except Exception as e:
        print(f"Error applying cartoon effect: {e}")
        return frame

def main():
    # Setup MediaPipe Face Stylizer
    try:
        with open(FACE_STYLIZER_MODEL_PATH, 'rb') as f:
            model_data = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceStylizerOptions(base_options=base_options)
        stylizer = vision.FaceStylizer.create_from_options(options)
    except Exception as e:
        print(f"Error loading Face Stylizer model: {e}")
        return

    # Video capture
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    if not cap.isOpened():
        print(f"Error opening video file {VIDEO_INPUT_PATH}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame extraction at specified time intervals
    for i, interval in enumerate(TIME_INTERVALS):
        frame_number = int(interval * fps)
        if frame_number >= total_frames:
            print(f"Skipping interval {interval}s: frame number {frame_number} exceeds total frames.")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame at {interval}s.")
            continue
        
        # Apply cartoon effect to frame
        cartoon_frame = apply_cartoon_effect(frame, stylizer)
        resized_frame = resize_and_show(cartoon_frame)

        # Save the processed frame as an image
        output_image_filename = f'cartoon_frame_{i + 1}.png'
        cv2.imwrite(output_image_filename, resized_frame)
        print(f"Saved {output_image_filename} at {interval}s")

    cap.release()
    print("Processing complete. Images saved.")

if __name__ == "__main__":
    main()
