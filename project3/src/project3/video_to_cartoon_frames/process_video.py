import cv2
import os
from apply_cartoon_effect import apply_cartoon_effect
from resize_and_show import resize_and_show

def process_video(video_input_path, face_stylizer_model_path, time_intervals, output_dir):
    """
    Process a video to capture frames at specified time intervals, apply a cartoon effect
    to these frames, and save the processed frames as images in the specified output directory.
    
    Args:
        video_input_path (str): Path to the input video file.
        face_stylizer_model_path (str): Path to the Face Stylizer model file.
        time_intervals (list of float): List of time intervals (in seconds) to capture frames.
        output_dir (str): Directory where the processed images will be saved.
    """
    try:
        # Load Face Stylizer model
        with open(face_stylizer_model_path, 'rb') as f:
            model_data = f.read()
        base_options = python.BaseOptions(model_asset_buffer=model_data)
        options = vision.FaceStylizerOptions(base_options=base_options)
        stylizer = vision.FaceStylizer.create_from_options(options)
    except Exception as e:
        print(f"Error loading Face Stylizer model: {e}")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Video capture
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame extraction at specified time intervals
    for i, interval in enumerate(time_intervals):
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

        # Save the processed frame as an image in the specified output directory
        output_image_filename = os.path.join(output_dir, f'cartoon_frame_{i + 1}.png')
        cv2.imwrite(output_image_filename, resized_frame)
        print(f"Saved {output_image_filename} at {interval}s")

    cap.release()
    print("Processing complete. Images saved.")
