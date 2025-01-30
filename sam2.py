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

    @staticmethod
    def __load_json(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def __process_frame(frame, masks, output_dir, frame_idx, bboxes, keypoints, adjusted_keypoints):
        for i, mask in enumerate(masks):
            x1, y1, x2, y2 = map(int, bboxes[i])  # Ensure bbox coordinates are integers
            cropped_frame = frame[y1:y2, x1:x2]
            if cropped_frame.size == 0:
                continue  # Skip if the cropped frame is empty
            mask_img = np.zeros((cropped_frame.shape[0], cropped_frame.shape[1], 4), dtype=np.uint8)
            mask_img[:, :, :3] = cropped_frame  # Copy the cropped frame to the mask image
            mask_img[:, :, 3] = 0  # Set alpha channel to 0 (transparent) for the whole image
            mask_img[mask[y1:y2, x1:x2]] = np.concatenate(
                (cropped_frame[mask[y1:y2, x1:x2]],
                 np.full((mask[y1:y2, x1:x2].sum(), 1),
                         255,
                         dtype=np.uint8)),
                axis=1)  # Set alpha channel to 255 (opaque) for the mask region

            output_path = os.path.join(output_dir, f'{frame_idx:04d}.png')
            cv2.imwrite(output_path, mask_img.astype(np.uint8))  # Ensure mask is a numpy array of type uint8

            # Adjust keypoints
            adjusted_kps = []
            for kp in keypoints[i]:
                adjusted_kps.append([int(kp[0] - x1), int(kp[1] - y1)])
            adjusted_keypoints.append(adjusted_kps)

    def run(self):
        # Load JSON data
        data_fig1 = self.__load_json(self.__json1_path)
        data_fig2 = self.__load_json(self.__json2_path)

        # Load a model
        model = SAM("sam2_b.pt")

        # Open video capture
        cap = cv2.VideoCapture(str(self.__video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        adjusted_keypoints_fig1 = []
        adjusted_keypoints_fig2 = []

        # Loop over frames with tqdm progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                bboxes_fig1 = []
                bboxes_fig2 = []
                keypoints_fig1 = []
                keypoints_fig2 = []

                if frame_idx < len(data_fig1):
                    boxes_fig1 = data_fig1[frame_idx]['box']
                    x1, y1, w, h, conf = boxes_fig1
                    x2, y2 = x1 + w, y1 + h
                    bboxes_fig1.append([x1, y1, x2, y2])
                    keypoints_fig1.append(data_fig1[frame_idx]['joints2d'])

                if frame_idx < len(data_fig2):
                    boxes_fig2 = data_fig2[frame_idx]['box']
                    x1, y1, w, h, conf = boxes_fig2
                    x2, y2 = x1 + w, y1 + h
                    bboxes_fig2.append([x1, y1, x2, y2])
                    keypoints_fig2.append(data_fig2[frame_idx]['joints2d'])

                # Process fig1
                if bboxes_fig1:
                    bboxes = np.array(bboxes_fig1)
                    results = model(frame, bboxes=bboxes, labels=[1]*len(bboxes_fig1), verbose=False)
                    for result in results:
                        masks = result.masks.data.cpu().numpy().astype(bool)
                        self.__process_frame(frame, masks, self.__output_dir_fig1, frame_idx, bboxes_fig1, keypoints_fig1, adjusted_keypoints_fig1)

                # Process fig2
                if bboxes_fig2:
                    bboxes = np.array(bboxes_fig2)
                    results = model(frame, bboxes=bboxes, labels=[1]*len(bboxes_fig2), verbose=False)
                    for result in results:
                        masks = result.masks.data.cpu().numpy().astype(bool)
                        self.__process_frame(frame, masks, self.__output_dir_fig2, frame_idx, bboxes_fig2, keypoints_fig2, adjusted_keypoints_fig2)

                pbar.update(1)

        cap.release()

        # Save adjusted keypoints to JSON files
        with open(os.path.join(self.__output_dir_fig1, '../fig1_poses2d.json'), 'w') as f:
            json.dump(adjusted_keypoints_fig1, f, indent=4)
        with open(os.path.join(self.__output_dir_fig2, '../fig2_poses2d.json'), 'w') as f:
            json.dump(adjusted_keypoints_fig2, f, indent=4)

        print(f"Masked frames saved to {self.__output_dir_fig1} and {self.__output_dir_fig2}")
        print(f"Adjusted keypoints saved to {os.path.join(self.__output_dir_fig1, '../fig1_poses2d.json')} and {os.path.join(self.__output_dir_fig2, '../fig2_poses2d.json')}")

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