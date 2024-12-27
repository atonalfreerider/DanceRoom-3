import cv2
import json
import numpy as np
from tqdm import tqdm
from ultralytics import SAM
import os
import argparse


class Sam2:
    def __init__(self, video_path, output_dir, json1_path, json2_path):
        self.__video_path = video_path
        self.__output_dir_fig1 = os.path.join(output_dir, 'fig1')
        self.__output_dir_fig2 = os.path.join(output_dir, 'fig2')
        self.__json1_path = json1_path
        self.__json2_path = json2_path
        os.makedirs(self.__output_dir_fig1, exist_ok=True)
        os.makedirs(self.__output_dir_fig2, exist_ok=True)

    def __get_frame_count(self):
        cap = cv2.VideoCapture(str(self.__video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def __load_json(self, json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    def __process_frame(self, frame, boxes, output_dir, frame_idx):
        for box in boxes:
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            mask = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
            mask[:, :, 3] = 0  # Set alpha channel to 0 (transparent) for the whole image
            mask[y:y+h, x:x+w, :3] = frame[y:y+h, x:x+w]
            mask[y:y+h, x:x+w, 3] = 255  # Set alpha channel to 255 (opaque) for the box region
            output_path = os.path.join(output_dir, f'{frame_idx}.png')
            cv2.imwrite(output_path, mask.astype(np.uint8))  # Ensure mask is a numpy array of type uint8

    def run(self):
        # Load JSON data
        data_fig1 = self.__load_json(self.__json1_path)
        data_fig2 = self.__load_json(self.__json2_path)

        # Get total frame count using OpenCV
        total_frames = self.__get_frame_count()

        # Open video capture
        cap = cv2.VideoCapture(str(self.__video_path))

        # Loop over frames with tqdm progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx < len(data_fig1):
                    boxes_fig1 = data_fig1[frame_idx]['boxes']
                    self.__process_frame(frame, [boxes_fig1], self.__output_dir_fig1, frame_idx)

                if frame_idx < len(data_fig2):
                    boxes_fig2 = data_fig2[frame_idx]['boxes']
                    self.__process_frame(frame, [boxes_fig2], self.__output_dir_fig2, frame_idx)

                pbar.update(1)

        cap.release()
        print(f"Masked frames saved to {self.__output_dir_fig1} and {self.__output_dir_fig2}")


def main():
    parser = argparse.ArgumentParser(description="Process video and JSON files to extract masked frames.")
    parser.add_argument('--video', required=True, help='Path to the input video file.')
    parser.add_argument('--json_folder', required=True, help='Path to the folder containing the JSON files.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory.')

    args = parser.parse_args()

    video_path = args.video
    json_folder = args.json_folder
    output_dir = args.output_dir

    json1_path = os.path.join(json_folder, 'figure1-pose2d-boxes.json')
    json2_path = os.path.join(json_folder, 'figure2-pose2d-boxes.json')

    sam2 = Sam2(video_path, output_dir, json1_path, json2_path)
    sam2.run()

if __name__ == "__main__":
    main()