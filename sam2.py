import cv2
from tqdm import tqdm
from ultralytics import SAM


class Sam2:
    def __init__(self, video_path, output_dir):
        self.__video_path = video_path
        self.__output_path = output_dir + '/masks.json'

    def __get_frame_count(self):
        cap = cv2.VideoCapture(str(self.__video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames

    def run(self):
        # Load a model
        model = SAM("sam2_b.pt")

        # Initialize an empty list to store all frame data
        frames_data = []

        # Get total frame count using OpenCV
        total_frames = self.__get_frame_count()

        # Run inference in a loop with stream=True
        results_generator = model.predict(source=str(self.__video_path), stream=True, verbose=False)

        # Loop over the generator with tqdm progress bar
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            for frame_idx, r in enumerate(results_generator):
                frame_contours = []  # List to store contours for the current frame

                if r.masks is not None:
                    # Loop over each mask
                    for contour in r.masks.xy:
                        contour_list = [[int(point[0]), int(point[1])] for point in contour]
                        frame_contours.append(contour_list)

                # Append the frame's contours to the frames_data list
                frames_data.append(frame_contours)

                # Update progress bar
                pbar.update(1)

        # Save the results to a JSON file
        utils.save_json(frames_data, self.__output_path)

        print(f"Mask contours saved to {self.__output_path}")