import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict, Any, Optional
from tqdm import tqdm

def __load_vo_data(output_dir: str) -> List[np.ndarray]:
    """Load visual odometry data from json"""
    # Construct VO json path
    vo_path = output_dir + '/' + [f for f in os.listdir(output_dir) if f.endswith('.vo.json')][0]

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

def __load_poses_data(output_dir: str) -> Any | None:
    """Load 3D poses data from json"""
    poses_path = output_dir + '/' + [f for f in os.listdir(output_dir) if f == 'poses3d.json'][0]
    
    try:
        with open(poses_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No poses data found at {poses_path}")
        return None

def __adjust_3d_points(points: List[List[float]], rotation_matrix: np.ndarray) -> List[List[float]]:
    """Apply rotation to 3D points"""
    points_array = np.array(points)

    # Apply rotation
    rotated_points = np.dot(points_array, rotation_matrix.T)
    
    # Restore the original shape
    rotated_points = np.expand_dims(rotated_points, axis=1)
    
    return rotated_points.tolist()

def __calculate_distance_to_camera(joints3d: List[List[float]]) -> float:
    """Calculate average distance of all joints to camera origin"""
    points = np.array(joints3d).reshape(-1, 3)

    # Calculate Euclidean distance for each point and take mean
    distances = np.sqrt(np.sum(points ** 2, axis=1))
    return float(np.mean(distances))

def __find_closest_match(current_pose: List[List[float]], previous_poses: List[List[float]]) -> int:
    """Find index of the pose in previous_poses that best matches current_pose"""
    if not previous_poses:
        return 0
        
    current = np.array(current_pose)
    min_dist = float('inf')
    best_match = 0
    
    for i, prev in enumerate(previous_poses):
        prev = np.array(prev)
        # Get minimum number of joints between the two poses
        min_joints = min(current.shape[0], prev.shape[0])
        # Calculate mean joint position difference using only common joints
        dist = np.mean(np.sqrt(np.sum((current[:min_joints] - prev[:min_joints]) ** 2, axis=1)))
        if dist < min_dist:
            min_dist = dist
            best_match = i
            
    return best_match

def __convert_joints_to_xyz_format(joints: List[List[float]], pad_with_t_pose: bool = True) -> List[Dict[str, float]]:
    """Convert joints array to list of x,y,z dictionaries"""
    converted_joints = []
    
    # Convert joints to numpy array for easier handling
    joints_array = np.array(joints).reshape(-1, 3)
    
    # Convert each joint
    for joint in joints_array:
        converted_joints.append({
            "x": float(joint[0]) / 1000.0,  # Convert mm to meters
            "y": -float(joint[1]) / 1000.0,  # Invert Y and convert to meters
            "z": float(joint[2]) / 1000.0   # Convert mm to meters
        })
    
    return converted_joints[:24]  # Ensure we only return 24 joints

def __interpolate_poses(poses: List[Optional[List[Dict[str, float]]]], desc: str = "") -> List[List[Dict[str, float]]]:
    """Interpolate missing poses and fill edges with T-pose if needed"""
    t_pose = __create_t_pose()
    interpolated_poses = []
    
    # Find first valid pose or use T-pose
    first_valid_idx = next((i for i, pose in enumerate(poses) if pose is not None), -1)
    
    last_valid_pose = poses[first_valid_idx]
    last_valid_idx = first_valid_idx
    interpolated_poses.append(last_valid_pose)
    
    # Process the rest of the poses
    for i in tqdm(range(first_valid_idx + 1, len(poses)), desc=f"Interpolating {desc}"):
        if poses[i] is not None:
            # If there were missing poses between last valid and current
            if i - last_valid_idx > 1:
                # Interpolate between last valid pose and current pose
                for j in range(1, i - last_valid_idx):
                    ratio = j / (i - last_valid_idx)
                    interpolated_pose = []
                    
                    # Interpolate all 24 joints
                    for joint_idx in range(24):
                        interpolated_joint = {}
                        for coord in ['x', 'y', 'z']:
                            start_val = last_valid_pose[joint_idx][coord]
                            end_val = poses[i][joint_idx][coord]
                            interpolated_joint[coord] = start_val + (end_val - start_val) * ratio
                        interpolated_pose.append(interpolated_joint)
                    
                    interpolated_poses.append(interpolated_pose)
            
            last_valid_pose = poses[i]
            last_valid_idx = i
            interpolated_poses.append(last_valid_pose)
        
    # Fill remaining poses with T-pose if needed
    while len(interpolated_poses) < len(poses):
        print("filling with T")
        interpolated_poses.append(t_pose)

    return interpolated_poses

def process_poses(output_dir: str):
    """Process and adjust 3D poses based on camera movement"""
    print("Loading data...")
    raw_rotations = __load_vo_data(output_dir)
    poses_data = __load_poses_data(output_dir)
    
    if not poses_data or not raw_rotations:
        print("poses or rotations missing")
        return
    
    # Get metadata
    metadata = poses_data['metadata']
    frames = poses_data['frames']
    total_frames = metadata['total_frames']
    
    # Analyze frame data
    print(f"\nTotal frames in poses3d.json: {len(frames)}")
    
    # Lists to store tracked figures
    figure1_frames = [None] * total_frames  # Pre-allocate with correct size
    figure2_frames = [None] * total_frames
    previous_figure1_joints = None
    previous_figure2_joints = None
    
    print("Processing frames...")
    for frame_num in tqdm(sorted(map(int, frames.keys())), desc="Processing frames"):
        frame_str = str(frame_num)
        frame_data = frames[frame_str]
        
        # Skip empty frames
        if not frame_data:
            print("empty frame")
            continue
        
        # Get rotation matrix for this frame
        if frame_num < len(raw_rotations):
            rotation_matrix = raw_rotations[frame_num]
            
            # Process all poses in this frame
            all_poses = []
            
            for person_data in frame_data:
                adjusted_joints = __adjust_3d_points(person_data['joints3d'], rotation_matrix)
                distance = __calculate_distance_to_camera(adjusted_joints)
                all_poses.append((adjusted_joints, distance))
            
            # Sort all poses by distance to camera
            all_poses.sort(key=lambda x: x[1])
            
            # Get the two closest poses if we have at least two poses
            if len(all_poses) >= 2:
                pose1, pose2 = all_poses[0][0], all_poses[1][0]
                
                # For first frame, simply assign poses
                if previous_figure1_joints is None:
                    previous_figure1_joints = pose1
                    previous_figure2_joints = pose2
                    figure1_frames[frame_num] = __convert_joints_to_xyz_format(pose1)
                    figure2_frames[frame_num] = __convert_joints_to_xyz_format(pose2)
                else:
                    # Find best matches for pose
                    prev_poses = [previous_figure1_joints, previous_figure2_joints]
                    
                    # Find closest match
                    match1 = __find_closest_match(pose1, prev_poses)
                    
                    # Update tracking
                    if match1 == 0:
                        figure1_frames[frame_num] = __convert_joints_to_xyz_format(pose1)
                        figure2_frames[frame_num] = __convert_joints_to_xyz_format(pose2)
                        previous_figure1_joints = pose1
                        previous_figure2_joints = pose2
                    else:
                        figure1_frames[frame_num] = __convert_joints_to_xyz_format(pose2)
                        figure2_frames[frame_num] = __convert_joints_to_xyz_format(pose1)
                        previous_figure1_joints = pose2
                        previous_figure2_joints = pose1
    
    print("Interpolating poses...")
    # Interpolate missing poses
    figure1_frames = __interpolate_poses(figure1_frames, "figure 1")
    figure2_frames = __interpolate_poses(figure2_frames, "figure 2")
    
    print("Saving results...")
    # Save figure1.json and figure2.json
    with open(os.path.join(output_dir, 'figure1.json'), 'w') as f:
        json.dump(figure1_frames, f, indent=2)
    
    with open(os.path.join(output_dir, 'figure2.json'), 'w') as f:
        json.dump(figure2_frames, f, indent=2)
    
    print(f"Figure tracking data saved to {output_dir}/figure1.json and {output_dir}/figure2.json")


def __create_t_pose() -> List[Dict[str, float]]:
    """Create a basic T-pose with 24 joints"""
    # Basic T-pose joint positions (in meters)
    t_pose = []

    # Torso and head (joints 0-3)
    t_pose.extend([
        {"x": 0.0, "y": 1.0, "z": 0.0},  # Hip
        {"x": 0.0, "y": 1.3, "z": 0.0},  # Spine
        {"x": 0.0, "y": 1.5, "z": 0.0},  # Chest
        {"x": 0.0, "y": 1.7, "z": 0.0},  # Head
    ])

    # Left arm (joints 4-7)
    t_pose.extend([
        {"x": -0.2, "y": 1.5, "z": 0.0},  # Left shoulder
        {"x": -0.5, "y": 1.5, "z": 0.0},  # Left elbow
        {"x": -0.7, "y": 1.5, "z": 0.0},  # Left wrist
        {"x": -0.8, "y": 1.5, "z": 0.0},  # Left hand
    ])

    # Right arm (joints 8-11)
    t_pose.extend([
        {"x": 0.2, "y": 1.5, "z": 0.0},  # Right shoulder
        {"x": 0.5, "y": 1.5, "z": 0.0},  # Right elbow
        {"x": 0.7, "y": 1.5, "z": 0.0},  # Right wrist
        {"x": 0.8, "y": 1.5, "z": 0.0},  # Right hand
    ])

    # Left leg (joints 12-15)
    t_pose.extend([
        {"x": -0.1, "y": 1.0, "z": 0.0},  # Left hip
        {"x": -0.1, "y": 0.6, "z": 0.0},  # Left knee
        {"x": -0.1, "y": 0.1, "z": 0.0},  # Left ankle
        {"x": -0.1, "y": 0.0, "z": 0.1},  # Left foot
    ])

    # Right leg (joints 16-19)
    t_pose.extend([
        {"x": 0.1, "y": 1.0, "z": 0.0},  # Right hip
        {"x": 0.1, "y": 0.6, "z": 0.0},  # Right knee
        {"x": 0.1, "y": 0.1, "z": 0.0},  # Right ankle
        {"x": 0.1, "y": 0.0, "z": 0.1},  # Right foot
    ])

    # Additional joints (20-23) - simplified positions
    t_pose.extend([
        {"x": 0.0, "y": 1.8, "z": 0.0},  # Nose
        {"x": -0.1, "y": 1.7, "z": 0.0},  # Left eye
        {"x": 0.1, "y": 1.7, "z": 0.0},  # Right eye
        {"x": 0.0, "y": 1.6, "z": 0.0},  # Neck
    ])

    return t_pose

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adjust 3D poses based on camera movement.")
    parser.add_argument("--output_dir", help="Path to the output directory")
    args = parser.parse_args()

    process_poses(args.output_dir)
