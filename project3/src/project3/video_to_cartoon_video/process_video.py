# project3/process_video.py
import cv2
from load_face_stylizer_model import load_face_stylizer_model
from apply_cartoon_effect import apply_cartoon_effect
from resize_and_show import resize_and_show

def process_video(input_video_path, output_video_path, model_path):
    """
    Processes the input video by applying a cartoon effect to each frame and saves the result to the output video file.

    Args:
        input_video_path (str): Path to the input video file.
        output_video_path (str): Path to the output video file.
        model_path (str): Path to the Face Stylizer model file.
    """
    # Setup MediaPipe Face Stylizer
    stylizer = load_face_stylizer_model(model_path)
    if stylizer is None:
        return

    # Video capture
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_seconds = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'XVID' for .avi files
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 60, (480, 480))  # Use 60 FPS and 480p resolution

    # Capture frames at the calculated intervals
    num_frames = int(duration_seconds * 60)
    for i in range(num_frames):
        frame_number = int(i * fps / 60)
        if frame_number >= int(cap.get(cv2.CAP_PROP_FRAME_COUNT)):
            print(f"Skipping frame {i + 1}: frame number {frame_number} exceeds total frames.")
            continue
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame number {frame_number}.")
            continue
        
        # Apply cartoon effect to frame
        cartoon_frame = apply_cartoon_effect(frame, stylizer)
        resized_frame = resize_and_show(cartoon_frame)

        # Write the processed frame to the video file
        video_writer.write(resized_frame)
        print(f"Processed and wrote frame {i + 1} at frame number {frame_number}")

    cap.release()
    video_writer.release()
    print(f"Processing complete. Video saved to {output_video_path}")