import sys
import os

from process_video import process_video

def main():
    """
    Main function to execute the video processing workflow.
    It checks the existence of the input video file and the Face Stylizer model,
    and if both are present, it processes the video to apply a cartoon effect.
    """
    # Define paths for the input video, output video, and model file
    input_video_path = r'project3\src\project3\video_to_cartoon_video\data\michal2.mp4'  # Path to the input video file
    output_video_path = r'project3\src\project3\video_to_cartoon_video\output\michal_processed.mp4'  # Path to save the processed video file
    model_path = r'project3\src\project3\video_to_cartoon_video\face_stylizer.task'  # Path to the Face Stylizer model file

    # Check if the input video file exists
    if not os.path.exists(input_video_path):
        print(f"Input video path does not exist: {input_video_path}")
        return

    # Check if the Face Stylizer model file exists
    if not os.path.exists(model_path):
        print(f"Model path does not exist: {model_path}")
        return

    # Process the video by applying cartoon effect and save the result
    process_video(input_video_path, output_video_path, model_path)

if __name__ == "__main__":
    main()
