import os
from process_video import process_video

def main():
    """
    Main function to execute the video processing workflow.
    
    It checks the existence of the input video file and the Face Stylizer model,
    and if both are present, it processes the video to apply a cartoon effect to
    frames captured at specified time intervals, saving the results in a defined output directory.
    """
    # Define paths for the input video, Face Stylizer model, and time intervals
    video_input_path = r'project3\src\project3\video_to_cartoon_frames\data\michal.mp4'
    face_stylizer_model_path = r'project3\src\project3\video_to_cartoon_frames\face_stylizer.task'
    output_dir = r'project3\src\project3\video_to_cartoon_frames\output'  # Directory to save processed images
    time_intervals = [0, 1.5, 3]  # Time intervals (in seconds) to capture frames

    # Check if the input video file exists
    if not os.path.exists(video_input_path):
        print(f"Input video path does not exist: {video_input_path}")
        return

    # Check if the Face Stylizer model file exists
    if not os.path.exists(face_stylizer_model_path):
        print(f"Model path does not exist: {face_stylizer_model_path}")
        return

    # Process the video by applying cartoon effect and save the result
    process_video(video_input_path, face_stylizer_model_path, time_intervals, output_dir)

# Entry point of the script
if __name__ == "__main__":
    main()
