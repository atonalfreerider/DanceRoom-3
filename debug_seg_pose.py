import cv2
import json
import os
import argparse

def load_json(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def draw_keypoints(image, keypoints):
    for kp in keypoints:
        x, y = kp
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)
    return image

def visualize_poses(fig_dir, json_path):
    keypoints_data = load_json(json_path)
    frame_files = sorted([f for f in os.listdir(fig_dir) if f.endswith('.png')])

    for frame_idx, frame_file in enumerate(frame_files):
        frame_path = os.path.join(fig_dir, frame_file)
        frame = cv2.imread(frame_path)
        keypoints = keypoints_data[frame_idx]

        frame_with_keypoints = draw_keypoints(frame, keypoints)
        cv2.imshow('Frame with Keypoints', frame_with_keypoints)

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description="Visualize 2D joints on segmented images.")
    parser.add_argument('--parent_dir', required=True, help='Path to the parent directory containing all inputs.')

    args = parser.parse_args()

    fig1_dir = os.path.join(args.parent_dir, 'fig1')
    fig1_json = os.path.join(args.parent_dir, 'fig1_poses2d.json')
    fig2_dir = os.path.join(args.parent_dir, 'fig2')
    fig2_json = os.path.join(args.parent_dir, 'fig2_poses2d.json')

    print("Visualizing Figure 1")
    visualize_poses(fig1_dir, fig1_json)

    print("Visualizing Figure 2")
    visualize_poses(fig2_dir, fig2_json)

if __name__ == "__main__":
    main()
