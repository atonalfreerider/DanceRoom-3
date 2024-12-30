import cv2
import json
import sys
import os
import shutil

def extract_video_metadata(video_path):
    # Check if the file exists
    if not os.path.isfile(video_path):
        print(f"Error: The file '{video_path}' does not exist.")
        sys.exit(1)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Cannot open the video file '{video_path}'.")
        sys.exit(1)

    # Retrieve frame count and FPS
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Handle cases where FPS might be zero
    if fps == 0:
        print("Error: FPS value is zero, cannot compute duration.")
        sys.exit(1)

    # Calculate duration
    duration = round(frame_count / fps, 3)

    # Release the video capture object
    video.release()

    return {
        "duration": duration,
        "frame_count": int(frame_count)
    }

def main():
    # Check if the video path is provided
    if len(sys.argv) != 3:
        print("Usage: python video_meta.py <path_to_video.mp4> <output_dir>")
        sys.exit(1)

    video_path = sys.argv[1]
    output_dir = sys.argv[2]
    metadata = extract_video_metadata(video_path)

    # Prepare the JSON file name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    json_filename = f"{base_name}_meta.json"
    json_filepath = os.path.join(os.path.dirname(video_path), json_filename)

    # Write metadata to JSON file
    with open(json_filepath, 'w') as json_file:
        json.dump(metadata, json_file, indent=4)

    # Move and rename the JSON file to the output directory
    shutil.move(json_filepath, os.path.join(output_dir, "video_meta.json"))

    print(f"Metadata has been written to '{os.path.join(output_dir, 'video_meta.json')}'.")

if __name__ == "__main__":
    main()
