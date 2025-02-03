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
        self.__json1_path = json1_path
        self.__json2_path = json2_path
        self.__output_dir = os.path.join(output_dir, 'seg')
        self.__masks_dir = os.path.join(output_dir, 'masks')
        self.__resized_dir = os.path.join(output_dir, 'resized_video')
        os.makedirs(self.__output_dir, exist_ok=True)
        os.makedirs(self.__masks_dir, exist_ok=True)
        os.makedirs(self.__resized_dir, exist_ok=True)
        self.__batch_size = 1000

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

    @staticmethod
    def __get_batch_folder(frame_idx, base_dir):
        batch_num = frame_idx // 1000
        batch_dir = os.path.join(base_dir, str(batch_num))
        os.makedirs(batch_dir, exist_ok=True)
        return batch_dir

    @staticmethod
    def __get_target_size(height, width):
        # Always return 480x640 (WxH), rotation will be handled separately
        return (480, 640)

    @staticmethod
    def __prepare_frame(frame, width, height):
        # Rotate frame if it's landscape
        if width > height:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        return frame

    @staticmethod
    def __resize_image(image, target_size):
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    def run(self):
        # Load JSON data
        data_fig1 = self.__load_json(self.__json1_path)
        data_fig2 = self.__load_json(self.__json2_path)

        # Load a model
        model = SAM("sam2_b.pt")

        # Open video capture
        cap = cv2.VideoCapture(str(self.__video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Loop over frames with tqdm progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame_idx in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Get target size and prepare frame
                target_size = self.__get_target_size(frame.shape[0], frame.shape[1])
                frame = self.__prepare_frame(frame, frame.shape[1], frame.shape[0])

                # Create resized frame
                resized_frame = self.__resize_image(frame, target_size)
                
                # Get batch folder and save
                batch_dir = self.__get_batch_folder(frame_idx, self.__resized_dir)
                resized_path = os.path.join(batch_dir, f"{frame_idx % 1000:03d}.jpg")
                cv2.imwrite(resized_path, resized_frame)

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

                masked_frame = np.zeros((frame.shape[0], frame.shape[1], 4), dtype=np.uint8)
                mask_binary = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)

                # Process fig1
                if bboxes_fig1:
                    results = model(frame, bboxes=np.array(bboxes_fig1), labels=[1]*len(bboxes_fig1), verbose=False)
                    for result in results:
                        masks = result.masks.data.cpu().numpy().astype(bool)
                        for mask in masks:
                            masked_frame[mask, :3] = frame[mask]  # copy BGR
                            masked_frame[mask, 3] = 255           # set alpha
                            mask_binary[mask] = 255

                # Process fig2
                if bboxes_fig2:
                    results = model(frame, bboxes=np.array(bboxes_fig2), labels=[1]*len(bboxes_fig2), verbose=False)
                    for result in results:
                        masks = result.masks.data.cpu().numpy().astype(bool)
                        for mask in masks:
                            masked_frame[mask, :3] = frame[mask]
                            masked_frame[mask, 3] = 255
                            mask_binary[mask] = 255

                # Save segmented output
                output_path = os.path.join(self.__output_dir, f"{frame_idx:04d}.png")
                cv2.imwrite(output_path, masked_frame)

                # Save mask output (resized)
                mask_resized = self.__resize_image(mask_binary, target_size)
                mask_path = os.path.join(self.__masks_dir, f"{frame_idx:04d}.png")
                cv2.imwrite(mask_path, mask_resized)

                pbar.update(1)

        cap.release()

        print(f"Processed frames saved to:")
        print(f"  Segmented: {self.__output_dir}")
        print(f"  Masks: {self.__masks_dir}")
        print(f"  Resized: {self.__resized_dir}")

def main():
    parser = argparse.ArgumentParser(description="Process video and JSON files to extract masked frames.")
    parser.add_argument('--video', required=True, help='Path to the input video file.')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory.')

    args = parser.parse_args()

    video_path = args.video
    output_dir = args.output_dir

    json1_path = os.path.join(output_dir, 'figure1-pose2d-boxes.json')
    json2_path = os.path.join(output_dir, 'figure2-pose2d-boxes.json')

    sam2 = Sam2(video_path, output_dir, json1_path, json2_path)
    sam2.run()

if __name__ == "__main__":
    main()