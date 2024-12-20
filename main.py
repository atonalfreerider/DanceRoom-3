import os
import json
import argparse
import numpy as np
from scipy.spatial.transform import Rotation
from typing import List, Dict, Any
from tqdm import tqdm
from collections import deque
from scipy.interpolate import interp1d

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
    poses_path = output_dir + '/' + [f for f in os.listdir(output_dir) if f.endswith('.poses3d.json')][0]
    
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

    # Flip the sign for all y points
    world_points[:, 1] = -world_points[:, 1]
    
    return world_points.tolist()

def __get_pose_center(joints3d: List[List[float]]) -> np.ndarray:
    """Get center position of pose (average of hip and spine joints)"""
    points = np.array(joints3d).reshape(-1, 3)
    # Use hip joint (index 0) for stable tracking, already in millimeters
    return points[0]

def __calculate_skeletal_height(joints3d: List[List[float]]) -> float:
    """Calculate height using SMPL joint structure"""
    points = np.array(joints3d).reshape(-1, 3)

    # Define segments using SMPL joint indices
    segments = [
        (0, 3),   # Pelvis to Spine1
        (3, 6),   # Spine1 to Spine2
        (6, 9),   # Spine2 to Spine3
        (9, 12),  # Spine3 to Neck
        (12, 15), # Neck to Head
        (0, 4),   # Pelvis to L_Knee
        (4, 7),   # L_Knee to L_Ankle
        (7, 10),  # L_Ankle to L_Foot
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

def __adjust_floor_level(poses1: List[List[Dict[str, float]]], poses2: List[List[Dict[str, float]]]) -> tuple[List[List[Dict[str, float]]], List[List[Dict[str, float]]]]:
    """Adjust poses so that the lowest point in each frame is at y=0"""
    adjusted_poses1 = []
    adjusted_poses2 = []
    
    # Indices for toe joints
    toe_indices = [10, 11]
    
    # Process each frame independently
    for frame1, frame2 in zip(poses1, poses2):
        # Find the lowest y value in this frame for both figures
        min_y1 = min(frame1[idx]['y'] for idx in toe_indices)
        min_y2 = min(frame2[idx]['y'] for idx in toe_indices)
        global_min_y = min(min_y1, min_y2)
        
        # Adjust all joints in this frame relative to the global minimum y
        adjusted_frame1 = [
            {'x': joint['x'], 
             'y': joint['y'] - global_min_y, 
             'z': joint['z']} 
            for joint in frame1
        ]
        adjusted_frame2 = [
            {'x': joint['x'], 
             'y': joint['y'] - global_min_y, 
             'z': joint['z']} 
            for joint in frame2
        ]
        
        adjusted_poses1.append(adjusted_frame1)
        adjusted_poses2.append(adjusted_frame2)
    
    return adjusted_poses1, adjusted_poses2

def __apply_moving_average(poses: List[List[Dict[str, float]]], window_size: int = 6) -> List[List[Dict[str, float]]]:
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

def __interpolate_missing_poses(valid_poses: List[List[Dict]], frame_indices: List[int], total_frames: int) -> List[List[Dict]]:
    """Interpolate missing poses using verified poses"""
    if not valid_poses:
        return []
        
    num_joints = len(valid_poses[0])  # Number of joints in pose
    interpolated_poses = []
    
    # Interpolate each joint separately
    for joint_idx in range(num_joints):
        # Prepare arrays for interpolation
        x_points = np.array(frame_indices)
        y_points = np.array([[pose[joint_idx]['x'], pose[joint_idx]['y'], pose[joint_idx]['z']] 
                           for pose in valid_poses])
        
        # Create interpolation function
        f = interp1d(x_points, y_points, axis=0, kind='linear', fill_value='extrapolate')
        
        # Generate complete sequence
        all_frames = np.arange(total_frames)
        interpolated = f(all_frames)
        
        # Store interpolated joint data
        if not interpolated_poses:
            interpolated_poses = [[{} for _ in range(num_joints)] for _ in range(total_frames)]
        
        # Fill in interpolated values
        for frame_idx, joint_pos in enumerate(interpolated):
            interpolated_poses[frame_idx][joint_idx] = {
                'x': float(joint_pos[0]),
                'y': float(joint_pos[1]),
                'z': float(joint_pos[2])
            }
    
    return interpolated_poses

def __get_pose_center_of_mass(pose: List[Dict[str, float]]) -> np.ndarray:
    """Calculate center of mass for a pose"""
    return np.mean([np.array([j['x'], j['y'], j['z']]) for j in pose], axis=0)

def __calculate_limb_lengths(pose: List[Dict[str, float]]) -> Dict[str, float]:
    """Calculate lengths of key limbs in a pose"""
    limb_pairs = {
        'spine': (0, 3),      # Pelvis to Spine1
        'torso': (3, 6),      # Spine1 to Spine2
        'chest': (6, 9),      # Spine2 to Spine3
        'neck': (9, 12),      # Spine3 to Neck
        'head': (12, 15),     # Neck to Head
        'left_thigh': (1, 4), # L_Hip to L_Knee
        'left_shin': (4, 7),  # L_Knee to L_Ankle
        'left_foot': (7, 10), # L_Ankle to L_Foot
        'right_thigh': (2, 5), # R_Hip to R_Knee
        'right_shin': (5, 8),  # R_Knee to R_Ankle
        'right_foot': (8, 11), # R_Ankle to R_Foot
        'left_upper_arm': (16, 18),  # L_Shoulder to L_Elbow
        'left_forearm': (18, 20),    # L_Elbow to L_Wrist
        'left_hand': (20, 22),       # L_Wrist to L_Hand
        'right_upper_arm': (17, 19),  # R_Shoulder to R_Elbow
        'right_forearm': (19, 21),    # R_Elbow to R_Wrist
        'right_hand': (21, 23)        # R_Wrist to R_Hand
    }
    
    lengths = {}
    for limb, (start_idx, end_idx) in limb_pairs.items():
        start = np.array([pose[start_idx]['x'], pose[start_idx]['y'], pose[start_idx]['z']])
        end = np.array([pose[end_idx]['x'], pose[end_idx]['y'], pose[end_idx]['z']])
        lengths[limb] = np.linalg.norm(end - start)
    
    return lengths

def __normalize_body_scale(poses: List[List[Dict[str, float]]]) -> List[List[Dict[str, float]]]:
    """Normalize body scale using median limb lengths"""
    if not poses:
        return poses
        
    # Calculate limb lengths for all frames
    all_lengths = [__calculate_limb_lengths(pose) for pose in poses]
    
    # Calculate median length for each limb
    median_lengths = {}
    for limb in all_lengths[0].keys():
        lengths = [frame_lengths[limb] for frame_lengths in all_lengths]
        median_lengths[limb] = np.median(lengths)
    
    # Calculate average scale factor across all limbs for each frame
    scale_factors = []
    for frame_lengths in all_lengths:
        ratios = [median_lengths[limb] / frame_lengths[limb] for limb in frame_lengths.keys()]
        scale_factors.append(np.median(ratios))
    
    # Apply scaling to each frame
    normalized_poses = []
    for frame_idx, pose in enumerate(poses):
        scale = scale_factors[frame_idx]
        # Calculate center of pose
        center = __get_pose_center_of_mass(pose)
        
        # Scale points relative to center
        normalized_frame = []
        for joint in pose:
            point = np.array([joint['x'], joint['y'], joint['z']])
            vector_to_center = point - center
            scaled_vector = vector_to_center * scale
            scaled_point = center + scaled_vector
            
            normalized_frame.append({
                'x': float(scaled_point[0]),
                'y': float(scaled_point[1]),
                'z': float(scaled_point[2])
            })
        
        normalized_poses.append(normalized_frame)
    
    return normalized_poses

def __correct_foot_slip(poses1: List[List[Dict[str, float]]], poses2: List[List[Dict[str, float]]]) -> tuple[List[List[Dict[str, float]]], List[List[Dict[str, float]]]]:
    """Correct foot slip by finding the most stationary foot across both poses and applying the same correction"""
    if not poses1 or not poses2:
        return poses1, poses2
    
    # Indices for foot joints
    left_foot_idx = 10   # L_Foot
    right_foot_idx = 11  # R_Foot
    
    corrected_poses1 = [poses1[0]]  # Start with first frame
    corrected_poses2 = [poses2[0]]  # Start with first frame
    
    for i in range(1, len(poses1)):
        # Get current poses and previously corrected poses
        current_pose1 = poses1[i]
        current_pose2 = poses2[i]
        prev_pose1 = corrected_poses1[-1]  # Use last corrected pose
        prev_pose2 = corrected_poses2[-1]  # Use last corrected pose
        
        # Extract only x,z coordinates for feet positions
        def get_xz(joint): return np.array([joint['x'], joint['z']])
        
        # Figure 1 feet
        left1_current = get_xz(current_pose1[left_foot_idx])
        right1_current = get_xz(current_pose1[right_foot_idx])
        left1_prev = get_xz(prev_pose1[left_foot_idx])
        right1_prev = get_xz(prev_pose1[right_foot_idx])
        
        # Figure 2 feet
        left2_current = get_xz(current_pose2[left_foot_idx])
        right2_current = get_xz(current_pose2[right_foot_idx])
        left2_prev = get_xz(prev_pose2[left_foot_idx])
        right2_prev = get_xz(prev_pose2[right_foot_idx])
        
        # Calculate movement vectors for all feet (x,z only)
        movements = [
            (left1_current - left1_prev),   # Figure 1 left foot
            (right1_current - right1_prev),  # Figure 1 right foot
            (left2_current - left2_prev),    # Figure 2 left foot
            (right2_current - right2_prev)   # Figure 2 right foot
        ]
        
        # Find the movement with smallest magnitude
        movement_magnitudes = [np.linalg.norm(m) for m in movements]
        min_movement_idx = np.argmin(movement_magnitudes)
        correction_xz = movements[min_movement_idx]
        
        # Create full correction vector (x,y,z) where y=0
        correction = np.array([correction_xz[0], 0, correction_xz[1]])
        
        # Apply the same correction to all joints in both poses
        corrected_frame1 = []
        corrected_frame2 = []
        
        for joint in current_pose1:
            point = np.array([joint['x'], joint['y'], joint['z']]) - correction
            corrected_frame1.append({
                'x': float(point[0]),
                'y': float(point[1]),
                'z': float(point[2])
            })
            
        for joint in current_pose2:
            point = np.array([joint['x'], joint['y'], joint['z']]) - correction
            corrected_frame2.append({
                'x': float(point[0]),
                'y': float(point[1]),
                'z': float(point[2])
            })
        
        corrected_poses1.append(corrected_frame1)
        corrected_poses2.append(corrected_frame2)
    
    return corrected_poses1, corrected_poses2

def __apply_shadow_smoothing(poses: List[List[Dict[str, float]]], window_size: int = 12) -> List[List[Dict[str, float]]]:
    """Apply moving average smoothing to the geometric center of each pose's shadow on the floor (XZ plane)"""
    if not poses:
        return poses
    
    num_frames = len(poses)
    smoothed_poses = [[] for _ in range(num_frames)]
    
    # Separate queues for x, z coordinates of the shadow center
    x_window = deque(maxlen=window_size)
    z_window = deque(maxlen=window_size)
    
    # Process all frames
    for frame_idx in range(num_frames):
        frame = poses[frame_idx]
        
        # Calculate the geometric center of the shadow (XZ plane)
        x_center = np.mean([joint['x'] for joint in frame])
        z_center = np.mean([joint['z'] for joint in frame])
        
        x_window.append(x_center)
        z_window.append(z_center)
        
        # Calculate moving averages
        smoothed_x_center = sum(x_window) / len(x_window)
        smoothed_z_center = sum(z_window) / len(z_window)
        
        # Calculate the translation needed to smooth the shadow
        translation_x = smoothed_x_center - x_center
        translation_z = smoothed_z_center - z_center
        
        # Apply the translation to all joints in the frame
        smoothed_frame = []
        for joint in frame:
            smoothed_frame.append({
                'x': joint['x'] + translation_x,
                'y': joint['y'],
                'z': joint['z'] + translation_z
            })
        
        smoothed_poses[frame_idx] = smoothed_frame
    
    return smoothed_poses

def process_poses(output_dir: str):
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
            # Store tuple of (joints, center, height, distance_to_camera)
            distance_to_camera = np.linalg.norm(center)
            valid_poses.append((adjusted_joints, center, height, distance_to_camera))
        
        # Ensure at least two poses
        if len(valid_poses) < 2:
            print(f"Warning: Frame {frame_num} has less than 2 poses, skipping")
            continue
            
        # First sort by distance to camera
        valid_poses.sort(key=lambda x: x[3])  # Sort by distance_to_camera
        closest_poses = valid_poses[:2]  # Take two closest poses
        
        # Then sort these two by height
        closest_poses.sort(key=lambda x: x[2], reverse=True)  # Sort by height (descending)
        
        # Refined mode with tracking logic
        pose1, pos1, height1, _ = closest_poses[0]  # Taller pose
        pose2, pos2, height2, _ = closest_poses[1]  # Shorter pose
        
        if previous_figure1_pos is None:
            # Initialize tracking with taller pose as figure1
            fig1_pose, fig2_pose = pose1, pose2
            previous_figure1_pos = pos1
            previous_figure1_height = height1
        else:
            # Use tracking logic but maintain height order
            valid1_to_fig1 = (__is_valid_movement(pos1, previous_figure1_pos) and 
                            __is_similar_height(height1, previous_figure1_height))
            valid2_to_fig1 = (__is_valid_movement(pos2, previous_figure1_pos) and 
                            __is_similar_height(height2, previous_figure1_height))
            
            if valid1_to_fig1 and height1 >= height2:
                fig1_pose, fig2_pose = pose1, pose2
                previous_figure1_pos = pos1
                previous_figure1_height = height1
            elif valid2_to_fig1 and height2 > height1:
                fig1_pose, fig2_pose = pose2, pose1
                previous_figure1_pos = pos2
                previous_figure1_height = height2
            else:
                # Default to height-based assignment if tracking fails
                fig1_pose, fig2_pose = pose1, pose2
                previous_figure1_pos = pos1
                previous_figure1_height = height1
        
        # Convert to xyz object format
        fig1_frame = [{"x": joint[0], "y": joint[1], "z": joint[2]} for joint in fig1_pose]
        fig2_frame = [{"x": joint[0], "y": joint[1], "z": joint[2]} for joint in fig2_pose]
        figure1_frames.append(fig1_frame)
        figure2_frames.append(fig2_frame)

    print("Processing with center tracking...")
    
    # Get frame ranges
    frame_numbers = sorted(map(int, frames.keys()))
    total_frames = len(frame_numbers)
    mid_idx = total_frames // 2
    
    # Initialize tracking structures
    valid_fig1_frames = []
    valid_fig2_frames = []
    valid_fig1_indices = []
    valid_fig2_indices = []
    valid_fig1_centers = []
    valid_fig2_centers = []
    
    # Track frames since last valid pose for radius expansion
    frames_since_fig1 = 0
    frames_since_fig2 = 0
    BASE_RADIUS = 0.5  # Base search radius in meters
    
    # Start from middle frame
    mid_frame = figure1_frames[mid_idx]
    mid_frame2 = figure2_frames[mid_idx]
    
    # Initialize tracking with middle frame
    valid_fig1_frames.append(mid_frame)
    valid_fig2_frames.append(mid_frame2)
    valid_fig1_indices.append(mid_idx)
    valid_fig2_indices.append(mid_idx)
    valid_fig1_centers.append(__get_pose_center_of_mass(mid_frame))
    valid_fig2_centers.append(__get_pose_center_of_mass(mid_frame2))
    
    # Process forward
    for idx in range(mid_idx + 1, len(figure1_frames)):
        curr_frame1 = figure1_frames[idx]
        curr_frame2 = figure2_frames[idx]
        
        center1 = __get_pose_center_of_mass(curr_frame1)
        center2 = __get_pose_center_of_mass(curr_frame2)
        
        # Calculate adaptive search radius
        radius1 = BASE_RADIUS + (frames_since_fig1 * 0.1)
        radius2 = BASE_RADIUS + (frames_since_fig2 * 0.1)
        
        # Check if movement is valid with adaptive radius
        dist1 = np.linalg.norm(center1 - valid_fig1_centers[-1])
        dist2 = np.linalg.norm(center2 - valid_fig2_centers[-1])
        
        if dist1 <= radius1:
            valid_fig1_frames.append(curr_frame1)
            valid_fig1_indices.append(idx)
            valid_fig1_centers.append(center1)
            frames_since_fig1 = 0  # Reset counter when pose found
        else:
            frames_since_fig1 += 1
        
        if dist2 <= radius2:
            valid_fig2_frames.append(curr_frame2)
            valid_fig2_indices.append(idx)
            valid_fig2_centers.append(center2)
            frames_since_fig2 = 0  # Reset counter when pose found
        else:
            frames_since_fig2 += 1
    
    # Reset counters for backward pass
    frames_since_fig1 = 0
    frames_since_fig2 = 0
    
    # Process backward
    for idx in range(mid_idx - 1, -1, -1):
        curr_frame1 = figure1_frames[idx]
        curr_frame2 = figure2_frames[idx]
        
        center1 = __get_pose_center_of_mass(curr_frame1)
        center2 = __get_pose_center_of_mass(curr_frame2)
        
        # Calculate adaptive search radius
        radius1 = BASE_RADIUS + (frames_since_fig1 * 0.1)
        radius2 = BASE_RADIUS + (frames_since_fig2 * 0.1)
        
        # Check if movement is valid with adaptive radius
        dist1 = np.linalg.norm(center1 - valid_fig1_centers[0])
        dist2 = np.linalg.norm(center2 - valid_fig2_centers[0])
        
        if dist1 <= radius1:
            valid_fig1_frames.insert(0, curr_frame1)
            valid_fig1_indices.insert(0, idx)
            valid_fig1_centers.insert(0, center1)
            frames_since_fig1 = 0
        else:
            frames_since_fig1 += 1
        
        if dist2 <= radius2:
            valid_fig2_frames.insert(0, curr_frame2)
            valid_fig2_indices.insert(0, idx)
            valid_fig2_centers.insert(0, center2)
            frames_since_fig2 = 0
        else:
            frames_since_fig2 += 1

    # Interpolate using gathered valid poses
    figure1_frames = __interpolate_missing_poses(valid_fig1_frames, valid_fig1_indices, total_frames)
    figure2_frames = __interpolate_missing_poses(valid_fig2_frames, valid_fig2_indices, total_frames)

    # Ensure we have valid data before proceeding
    if not figure1_frames or not figure2_frames:
        print("Error: No valid poses found after processing")
        return

    # Normalize body scale for both figures
    print("Normalizing body scale...")
    figure1_frames = __normalize_body_scale(figure1_frames)
    figure2_frames = __normalize_body_scale(figure2_frames)
    
    # Correct foot slip for both figures together
    print("Correcting foot slip...")
    figure1_frames, figure2_frames = __correct_foot_slip(figure1_frames, figure2_frames)

    # Apply floor adjustment
    figure1_frames, figure2_frames = __adjust_floor_level(figure1_frames, figure2_frames)
    
    # Apply shadow smoothing
    figure1_frames = __apply_shadow_smoothing(figure1_frames)
    figure2_frames = __apply_shadow_smoothing(figure2_frames)
    
    # Apply joint by joint moving average
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
    args = parser.parse_args()

    process_poses(args.output_dir)
