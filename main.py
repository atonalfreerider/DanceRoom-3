import os
import json
import argparse
from scipy.spatial.transform import Rotation, Slerp

def __load_vo_data(output_dir:str):
    """Load visual odometry data from json"""
    # Construct VO json path
    vo_path = output_dir + '/' + [f for f in os.listdir(output_dir) if f.endswith('.vo.json')][0]

    vo_data = []
    try:
        with open(vo_path, 'r') as f:
            vo_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: No VO data found at {vo_path}")
        return []

    # Get initial camera state
    initial_rotation = Rotation.from_quat([0, 1, 0, 0])

    # Process each frame
    raw_rotations = []  # Store raw rotation matrices for smoothing

    # calculate raw rotations
    for frame_data in vo_data:
        vo_rotation = Rotation.from_quat(frame_data[3:])
        relative_rotation = initial_rotation * vo_rotation.inv()
        raw_rotations.append(relative_rotation.as_matrix())

    return raw_rotations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Move 3D poses from camera space to world space.")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()

    raw_rotations = __load_vo_data(args.output_dir)
