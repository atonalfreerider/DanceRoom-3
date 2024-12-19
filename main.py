import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict, Any
from tqdm import tqdm
from collections import deque

def __load_vo_data(output_dir: str) -> List[np.ndarray]:
    """Load visual odometry data from json"""
    vo_path = output_dir + '/' + [f for f in os.listdir(output_dir) if f.endswith('.vo.json')][0]

    try:
        with open(vo_path, 'r') as f:
            vo_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: No VO data found at {vo_path}")
        return []

    # Process each frame
    rotations = []
    initial_rotation = Rotation.from_quat([0, 1, 0, 0])

    for frame_data in vo_data:
        # VO data contains absolute camera orientation
        camera_rotation = Rotation.from_quat(frame_data[3:])
        # Calculate world-to-camera transform
        world_to_camera = camera_rotation * initial_rotation
        # Store camera-to-world transform to convert points
        rotations.append(world_to_camera.inv().as_matrix())

    return rotations

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
    """Transform points from camera space to world space"""
    # Points are in millimeters, convert to meters
    points_array = np.array(points).reshape(-1, 3) / 1000.0

    # Transform points from camera space to world space
    world_points = np.dot(points_array, rotation_matrix)
    
    # Convert back to millimeters for consistency with rest of the pipeline
    world_points = world_points * 1000.0
    
    return world_points.tolist()

def __get_pose_center(joints3d: List[List[float]]) -> np.ndarray:
    """Get center position of pose (average of hip and spine joints)"""
    points = np.array(joints3d).reshape(-1, 3)
    # Use hip joint (index 0) for stable tracking, already in millimeters
    return points[0] / 1000.0  # Convert to meters for distance calculations

def __calculate_skeletal_height(joints3d: List[List[float]]) -> float:
    """Calculate height by summing leg and torso segments"""
    points = np.array(joints3d).reshape(-1, 3) / 1000.0  # Convert to meters

    # Define segments to measure (using COCO joint indices)
    segments = [
        (0, 1),  # hip to spine
        (1, 2),  # spine to chest
        (2, 23),  # chest to neck
        (23, 3),  # neck to head
        (0, 13),  # hip to knee
        (13, 14),  # knee to ankle
    ]

    total_height = 0
    for start_idx, end_idx in segments:
        segment_length = np.linalg.norm(points[end_idx] - points[start_idx])
        total_height += segment_length

    return float(total_height)

def __is_valid_movement(current_pos: np.ndarray, previous_pos: np.ndarray) -> bool:
    """Check if movement between frames is within threshold"""
    MAX_MOVEMENT = 0.5  # meters
    if previous_pos is None:
        return True
    distance = np.linalg.norm(current_pos - previous_pos)
    return distance <= MAX_MOVEMENT

def __is_similar_height(height1: float, height2: float, threshold: float = 0.15) -> bool:
    """Check if two heights are similar within threshold"""
    if height2 is None:
        return True
    return abs(height1 - height2) <= threshold

def __adjust_floor_level(poses: List[List[Dict[str, float]]]) -> List[List[Dict[str, float]]]:
    """Adjust poses so that the lowest point in each frame is at y=0"""
    adjusted_poses = []
    
    # Process each frame independently
    for frame in poses:
        # Find the lowest y value in this frame
        min_y = min(joint['y'] for joint in frame)
        
        # Adjust all joints in this frame relative to its minimum y
        adjusted_frame = [
            {'x': joint['x'], 
             'y': joint['y'] - min_y, 
             'z': joint['z']} 
            for joint in frame
        ]
        adjusted_poses.append(adjusted_frame)
    
    return adjusted_poses

def __apply_moving_average(poses: List[List[Dict[str, float]]], window_size: int = 10) -> List[List[Dict[str, float]]]:
    """Apply moving average smoothing to each joint separately"""
    if not poses:
        return poses
    
    num_frames = len(poses)
    num_joints = len(poses[0])
    smoothed_poses = [[] for _ in range(num_frames)]
    
    # Process each joint separately
    for joint_idx in range(num_joints):
        # Separate queues for x, y, z coordinates
        x_window = deque(maxlen=window_size)
        y_window = deque(maxlen=window_size)
        z_window = deque(maxlen=window_size)
        
        # Process all frames for this joint
        for frame_idx in range(num_frames):
            joint = poses[frame_idx][joint_idx]
            
            x_window.append(joint['x'])
            y_window.append(joint['y'])
            z_window.append(joint['z'])
            
            # Calculate moving averages
            smoothed_joint = {
                'x': sum(x_window) / len(x_window),
                'y': sum(y_window) / len(y_window),
                'z': sum(z_window) / len(z_window)
            }
            
            smoothed_poses[frame_idx].append(smoothed_joint)
    
    return smoothed_poses

def process_poses(output_dir: str, refine: bool = False):
    """Process and adjust 3D poses based on camera movement"""
    print("Loading data...")
    raw_rotations = __load_vo_data(output_dir)
    poses_data = __load_poses_data(output_dir)
    
    if not poses_data or not raw_rotations:
        print("poses or rotations missing")
        return
    
    # Get metadata
    frames = poses_data['frames']
    
    print(f"\nTotal frames in poses3d.json: {len(frames)}")
    
    # Lists to store tracked figures
    figure1_frames = []
    figure2_frames = []
    previous_figure1_pos = None
    previous_figure1_height = None
    
    print("Processing frames...")
    for frame_num in tqdm(sorted(map(int, frames.keys())), desc="Processing frames"):
        frame_str = str(frame_num)
        frame_data = frames[frame_str]
        
        if not frame_data or frame_num >= len(raw_rotations):
            continue
            
        rotation_matrix = raw_rotations[frame_num]
        valid_poses = []
        
        # Process all poses in frame
        for person_data in frame_data:
            adjusted_joints = __adjust_3d_points(person_data['joints3d'], rotation_matrix)
            center = __get_pose_center(adjusted_joints)
            height = __calculate_skeletal_height(adjusted_joints)
            valid_poses.append((adjusted_joints, center, height))
        
        # Ensure at least two poses
        if len(valid_poses) < 2:
            print(f"Warning: Frame {frame_num} has less than 2 poses, skipping")
            continue
            
        # Sort by distance to camera (using center position)
        valid_poses.sort(key=lambda x: np.linalg.norm(x[1]))
        
        # Keep only the two closest poses
        valid_poses = valid_poses[:2]
        
        if not refine:
            # Simple mode: always take the two closest poses
            pose1, _, _ = valid_poses[0]
            pose2, _, _ = valid_poses[1]
            # Convert to xyz object format
            figure1_frame = [{"x": joint[0]/1000.0, "y": -joint[1]/1000.0, "z": joint[2]/1000.0} for joint in pose1]
            figure2_frame = [{"x": joint[0]/1000.0, "y": -joint[1]/1000.0, "z": joint[2]/1000.0} for joint in pose2]
            figure1_frames.append(figure1_frame)
            figure2_frames.append(figure2_frame)
        else:
            # Refined mode with tracking logic
            pose1, pos1, height1 = valid_poses[0]
            pose2, pos2, height2 = valid_poses[1]
            
            if previous_figure1_pos is None:
                # Sort by height to ensure taller person is figure1
                if height1 >= height2:
                    fig1_pose, fig2_pose = pose1, pose2
                    previous_figure1_pos= pos1
                    previous_figure1_height = height1
                else:
                    fig1_pose, fig2_pose = pose2, pose1
                    previous_figure1_pos = pos2
                    previous_figure1_height = height2
            else:
                # Use tracking logic from original implementation
                valid1_to_fig1 = (__is_valid_movement(pos1, previous_figure1_pos) and 
                                __is_similar_height(height1, previous_figure1_height))
                valid2_to_fig1 = (__is_valid_movement(pos2, previous_figure1_pos) and 
                                __is_similar_height(height2, previous_figure1_height))
                
                if valid1_to_fig1:
                    fig1_pose, fig2_pose = pose1, pose2
                    previous_figure1_pos = pos1
                    previous_figure1_height = height1
                elif valid2_to_fig1:
                    fig1_pose, fig2_pose = pose2, pose1
                    previous_figure1_pos = pos2
                    previous_figure1_height = height2
                else:
                    # Default to distance-based assignment if tracking fails
                    fig1_pose, fig2_pose = pose1, pose2
                    previous_figure1_pos = pos1
                    previous_figure1_height = height1
            
            # Convert to xyz object format
            fig1_frame = [{"x": joint[0]/1000.0, "y": -joint[1]/1000.0, "z": joint[2]/1000.0} for joint in fig1_pose]
            fig2_frame = [{"x": joint[0]/1000.0, "y": -joint[1]/1000.0, "z": joint[2]/1000.0} for joint in fig2_pose]
            figure1_frames.append(fig1_frame)
            figure2_frames.append(fig2_frame)

    # Apply floor adjustment and smoothing before saving
    figure1_frames = __adjust_floor_level(figure1_frames)
    figure2_frames = __adjust_floor_level(figure2_frames)
    
    figure1_frames = __apply_moving_average(figure1_frames)
    figure2_frames = __apply_moving_average(figure2_frames)

    print("Saving results...")
    # Save figure1.json and figure2.json with the new format
    with open(os.path.join(output_dir, 'figure1.json'), 'w') as f:
        json.dump(figure1_frames, f, indent=2)
    
    with open(os.path.join(output_dir, 'figure2.json'), 'w') as f:
        json.dump(figure2_frames, f, indent=2)
    
    print(f"Figure tracking data saved to {output_dir}/figure1.json and {output_dir}/figure2.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adjust 3D poses based on camera movement.")
    parser.add_argument("--output_dir", help="Path to the output directory", required=True)
    parser.add_argument("--refine", action="store_true", help="Enable pose refinement and tracking")
    args = parser.parse_args()

    process_poses(args.output_dir, args.refine)
