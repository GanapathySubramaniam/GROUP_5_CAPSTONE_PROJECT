import cv2
import numpy as np

def combine_mp4_videos(video_list, output_path):
    """
    Combines a list of MP4 videos into a single video using OpenCV.
    
    :param video_list: List of paths to the input MP4 video files
    :param output_path: Path where the combined video will be saved
    """
    # Get video properties from the first video
    first_video = cv2.VideoCapture(video_list[0])
    fps = first_video.get(cv2.CAP_PROP_FPS)
    width = int(first_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_video.release()

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each video
    for video_path in video_list:
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        cap.release()

    # Release the output video writer
    out.release()

# Example usage:
# video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
# combine_mp4_videos(video_files, "combined_output.mp4")
