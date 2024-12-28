import cv2
import json
import argparse

def draw_pose_and_box(frame, joints2d, box, color):
    # Draw joints
    for joint in joints2d:
        cv2.circle(frame, (int(joint[0]), int(joint[1])), 3, color, -1)
    
    # Draw box
    x, y, w, h, _ = box
    cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

def process_video(video_path, output_path, pose_data_dir):
    # Load pose data for figure1 and figure2
    with open(f'{pose_data_dir}/figure1-pose2d-boxes.json', 'r') as f:
        figure1_data = json.load(f)
    with open(f'{pose_data_dir}/figure2-pose2d-boxes.json', 'r') as f:
        figure2_data = json.load(f)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(figure1_data):
            figure1 = figure1_data[frame_idx]
            draw_pose_and_box(frame, figure1['joints2d'], figure1['boxes'], (0, 0, 255))  # Red for figure1
        
        if frame_idx < len(figure2_data):
            figure2 = figure2_data[frame_idx]
            draw_pose_and_box(frame, figure2['joints2d'], figure2['boxes'], (255, 255, 255))  # White for figure2
        
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description='Draw 2D poses and boxes on video frames.')
    parser.add_argument('--video_path', type=str, required=True, help='Path to the input video file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to the output video file.')
    parser.add_argument('--pose_data_dir', type=str, required=True, help='Path to the directory containing 2D pose data JSON files.')
    
    args = parser.parse_args()
    process_video(args.video_path, args.output_path, args.pose_data_dir)

if __name__ == '__main__':
    main()