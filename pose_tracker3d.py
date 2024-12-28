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
    poses_path = output_dir + '/poses3d.json'
    
    try:
        with open(poses_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No poses data found at {poses_path}")
        return None

def __adjust_3d_points(person_data, rotation_matrix: np.ndarray):
    """Transform points from camera space to world space"""
    # Points are in millimeters, convert to meters
    points = person_data['joints3d']
    points_array = np.array(points).reshape(-1, 3) / 1000.0

    # Transform points from camera space to world space
    world_points = np.dot(points_array, rotation_matrix)

    # Flip the sign for all y points
    world_points[:, 1] = -world_points[:, 1]
    person_data['joints3d'] = world_points.tolist()
    
    return person_data

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

def __adjust_floor_level(poses1: List[Dict], poses2: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Adjust poses so that the lowest point in each frame is at y=0"""
    adjusted_poses1 = []
    adjusted_poses2 = []
    
    # Indices for toe joints
    toe_indices = [10, 11]
    
    # Process each frame independently
    for pose1, pose2 in zip(poses1, poses2):
        # Convert to numpy arrays
        joints3d1 = np.array(pose1['joints3d'])
        joints3d2 = np.array(pose2['joints3d'])
        
        # Find lowest y value (index 1 is y coordinate)
        min_y1 = min(joints3d1[toe_indices, 1])
        min_y2 = min(joints3d2[toe_indices, 1])
        global_min_y = min(min_y1, min_y2)
        
        # Create copies of original poses
        adjusted_pose1 = dict(pose1)
        adjusted_pose2 = dict(pose2)
        
        # Adjust y coordinates
        joints3d1[:, 1] -= global_min_y
        joints3d2[:, 1] -= global_min_y
        
        # Store adjusted joints
        adjusted_pose1['joints3d'] = joints3d1.tolist()
        adjusted_pose2['joints3d'] = joints3d2.tolist()
        
        adjusted_poses1.append(adjusted_pose1)
        adjusted_poses2.append(adjusted_pose2)
    
    return adjusted_poses1, adjusted_poses2

def __apply_moving_average(poses: List[Dict], window_size: int = 6) -> List[Dict]:
    """Apply moving average smoothing to each joint separately"""
    if not poses:
        return poses
    
    num_frames = len(poses)
    num_joints = len(poses[0]['joints3d'])
    smoothed_poses = []
    
    # Convert all joints3d to numpy arrays
    joints_data = np.array([pose['joints3d'] for pose in poses])
    
    # Create windows for each coordinate of each joint
    windows = np.zeros((num_joints, 3, window_size))  # (joint, xyz, window)
    smoothed_joints = np.zeros_like(joints_data)
    
    # Process each frame
    for frame_idx in range(num_frames):
        # Update windows with current frame data
        curr_joints = joints_data[frame_idx]
        for joint_idx in range(num_joints):
            for coord_idx in range(3):  # x, y, z
                # Roll window and add new value
                windows[joint_idx, coord_idx] = np.roll(windows[joint_idx, coord_idx], -1)
                windows[joint_idx, coord_idx, -1] = curr_joints[joint_idx, coord_idx]
                
                # Calculate moving average
                valid_window = windows[joint_idx, coord_idx, :min(window_size, frame_idx + 1)]
                if len(valid_window) > 0:
                    smoothed_joints[frame_idx, joint_idx, coord_idx] = np.mean(valid_window)
        
        # Create new pose with smoothed joints3d
        smoothed_pose = dict(poses[frame_idx])  # Copy original pose
        smoothed_pose['joints3d'] = smoothed_joints[frame_idx].tolist()
        smoothed_poses.append(smoothed_pose)
    
    return smoothed_poses

def __lerp_vectors(start_vec, end_vec, t):
    """Linear interpolation between two vectors"""
    return start_vec + (end_vec - start_vec) * t

def __interpolate_missing_poses(valid_poses: List[Dict], frame_indices: List[int], total_frames: int) -> List[Dict]:
    """Interpolate missing poses using verified poses"""
    if not valid_poses:
        return []

    # Use first valid pose as template
    template_pose = valid_poses[0]
    joints3d_data = [np.array(pose['joints3d'], dtype=np.float32) for pose in valid_poses]
    frame_indices = np.array(frame_indices, dtype=np.int32)
    
    interpolated_poses = []
    
    # Process each frame
    for frame_idx in range(total_frames):
        # Find surrounding keyframes
        next_idx = np.searchsorted(frame_indices, frame_idx)
        
        if next_idx == 0:
            # Before first keyframe - use first pose
            interpolated_joints = joints3d_data[0]
        elif next_idx >= len(frame_indices):
            # After last keyframe - use last pose
            interpolated_joints = joints3d_data[-1]
        else:
            # Interpolate between surrounding keyframes
            prev_idx = next_idx - 1
            prev_frame = frame_indices[prev_idx]
            next_frame = frame_indices[next_idx]
            t = (frame_idx - prev_frame) / (next_frame - prev_frame)
            
            interpolated_joints = __lerp_vectors(
                joints3d_data[prev_idx],
                joints3d_data[next_idx],
                t
            )
        
        # Create new pose with interpolated joints3d
        interpolated_pose = dict(template_pose)  # Preserve original data
        interpolated_pose['joints3d'] = interpolated_joints.tolist()
        interpolated_poses.append(interpolated_pose)
    
    return interpolated_poses

def __get_pose_center_of_mass(pose) -> np.ndarray:
    """Calculate center of mass for a pose"""
    return np.mean([np.array([j[0], j[1], j[2]]) for j in pose], axis=0)

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

def __normalize_body_scale(poses: List[Dict]) -> List[Dict]:
    """Normalize body scale using median limb lengths while preserving other data"""
    if not poses:
        return poses
        
    # Calculate limb lengths for all frames using joints3d
    all_lengths = []
    for pose in poses:
        joints3d = pose['joints3d']
        old_format = [{'x': j[0], 'y': j[1], 'z': j[2]} for j in joints3d]
        all_lengths.append(__calculate_limb_lengths(old_format))
    
    # Calculate scale factors
    scale_factors = []
    median_lengths = {}
    for limb in all_lengths[0].keys():
        lengths = [frame_lengths[limb] for frame_lengths in all_lengths]
        median_lengths[limb] = np.median(lengths)
    
    for frame_lengths in all_lengths:
        ratios = [median_lengths[limb] / frame_lengths[limb] for limb in frame_lengths.keys()]
        scale_factors.append(np.median(ratios))
    
    # Apply scaling to each frame's joints3d only
    normalized_poses = []
    for frame_idx, pose in enumerate(poses):
        scale = scale_factors[frame_idx]
        joints3d = np.array(pose['joints3d'])
        
        # Calculate center of pose
        center = np.mean(joints3d, axis=0)
        
        # Scale points relative to center
        vector_to_center = joints3d - center
        scaled_vectors = vector_to_center * scale
        scaled_joints3d = center + scaled_vectors
        
        # Create new pose with scaled joints3d but original other data
        normalized_pose = dict(pose)  # Make a copy of original pose
        normalized_pose['joints3d'] = scaled_joints3d.tolist()  # Only update joints3d
        normalized_poses.append(normalized_pose)
    
    return normalized_poses

def __correct_foot_slip(poses1: List[Dict], poses2: List[Dict]) -> tuple[List[Dict], List[Dict]]:
    """Correct foot slip while preserving all pose data"""
    if not poses1 or not poses2:
        return poses1, poses2
    
    # Start with copies of first frames
    corrected_poses1 = [dict(poses1[0])]
    corrected_poses2 = [dict(poses2[0])]
    
    # Indices for foot joints
    left_foot_idx = 10
    right_foot_idx = 11
    
    for i in range(1, len(poses1)):
        # Get current and previous poses
        current_pose1 = poses1[i]
        current_pose2 = poses2[i]
        prev_pose1 = corrected_poses1[-1]
        prev_pose2 = corrected_poses2[-1]
        
        def get_xz(joints3d, idx): 
            return np.array([joints3d[idx][0], joints3d[idx][2]])
        
        # Calculate feet movements using joints3d
        movements = [
            get_xz(current_pose1['joints3d'], left_foot_idx) - get_xz(prev_pose1['joints3d'], left_foot_idx),
            get_xz(current_pose1['joints3d'], right_foot_idx) - get_xz(prev_pose1['joints3d'], right_foot_idx),
            get_xz(current_pose2['joints3d'], left_foot_idx) - get_xz(prev_pose2['joints3d'], left_foot_idx),
            get_xz(current_pose2['joints3d'], right_foot_idx) - get_xz(prev_pose2['joints3d'], right_foot_idx)
        ]
        
        # Find minimum movement
        movement_magnitudes = [np.linalg.norm(m) for m in movements]
        min_movement_idx = np.argmin(movement_magnitudes)
        correction_xz = movements[min_movement_idx]
        
        # Create full correction vector
        correction = np.array([correction_xz[0], 0, correction_xz[1]])
        
        # Create new poses with corrected joints3d but original other data
        corrected_pose1 = dict(current_pose1)
        corrected_pose2 = dict(current_pose2)
        
        corrected_pose1['joints3d'] = (np.array(current_pose1['joints3d']) - correction).tolist()
        corrected_pose2['joints3d'] = (np.array(current_pose2['joints3d']) - correction).tolist()
        
        corrected_poses1.append(corrected_pose1)
        corrected_poses2.append(corrected_pose2)
    
    return corrected_poses1, corrected_poses2

def __apply_shadow_smoothing(poses: List[Dict], window_size: int = 12) -> List[Dict]:
    """Apply shadow smoothing while preserving all pose data"""
    if not poses:
        return poses
    
    num_frames = len(poses)
    smoothed_poses = []
    
    x_window = deque(maxlen=window_size)
    z_window = deque(maxlen=window_size)
    
    for frame_idx in range(num_frames):
        joints3d = np.array(poses[frame_idx]['joints3d'])
        
        x_center = np.mean(joints3d[:, 0])
        z_center = np.mean(joints3d[:, 2])
        
        x_window.append(x_center)
        z_window.append(z_center)
        
        smoothed_x_center = sum(x_window) / len(x_window)
        smoothed_z_center = sum(z_window) / len(z_window)
        
        translation = np.array([smoothed_x_center - x_center, 0, smoothed_z_center - z_center])
        smoothed_joints3d = joints3d + translation
        
        # Create new pose with smoothed joints3d but original other data
        smoothed_pose = dict(poses[frame_idx])  # Make a copy of original pose
        smoothed_pose['joints3d'] = smoothed_joints3d.tolist()  # Only update joints3d
        
        smoothed_poses.append(smoothed_pose)
    
    return smoothed_poses

def __convert_joints3d_to_xyz_list(joints3d):
    """Convert joints3d array format to list of xyz dictionaries"""
    return [{'x': joint[0], 'y': joint[1], 'z': joint[2]} for joint in joints3d]

def process_poses(output_dir: str):
    """Process and adjust 3D poses based on camera movement, and track lead and follow dancer"""
    print("Loading pose and camera rotation data...")
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
    figure1_pose2d_boxes = []
    figure2_pose2d_boxes = []
    previous_figure1_pos = None
    previous_figure1_height = None
    
    print("Extracting lead and follow and rotating all poses...")
    for frame_num in tqdm(sorted(map(int, frames.keys())), desc="Extracting lead and follow and applying rotations"):
        frame_str = str(frame_num)
        frame_data = frames[frame_str]
        
        if not frame_data or frame_num >= len(raw_rotations):
            continue
            
        rotation_matrix = raw_rotations[frame_num]
        valid_poses = []
        
        # Rotate all poses and extract tallest and closest poses
        for person_data in frame_data:
            adjusted_person_data = __adjust_3d_points(person_data, rotation_matrix)
            adjusted_joints = adjusted_person_data['joints3d']
            center = __get_pose_center(adjusted_joints)
            height = __calculate_skeletal_height(adjusted_joints)
            # Store tuple of (adjusted_person_data, center, height, distance_to_camera, original_data)
            distance_to_camera = np.linalg.norm(center)
            valid_poses.append((adjusted_person_data, center, height, distance_to_camera, person_data))
        
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
        pose1, pos1, height1, _, original_data1 = closest_poses[0]  # Taller pose
        pose2, pos2, height2, _, original_data2 = closest_poses[1]  # Shorter pose
        
        if previous_figure1_pos is None:
            # Initialize tracking with taller pose as figure1
            fig1_pose, fig2_pose = pose1, pose2
            fig1_original, fig2_original = original_data1, original_data2
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
                fig1_original, fig2_original = original_data1, original_data2
                previous_figure1_pos = pos1
                previous_figure1_height = height1
            elif valid2_to_fig1 and height2 > height1:
                fig1_pose, fig2_pose = pose2, pose1
                fig1_original, fig2_original = original_data2, original_data1
                previous_figure1_pos = pos2
                previous_figure1_height = height2
            else:
                # Default to height-based assignment if tracking fails
                fig1_pose, fig2_pose = pose1, pose2
                fig1_original, fig2_original = original_data1, original_data2
                previous_figure1_pos = pos1
                previous_figure1_height = height1

        figure1_frames.append(fig1_pose)
        figure2_frames.append(fig2_pose)
        figure1_pose2d_boxes.append({'joints2d': fig1_original['joints2d'], 'boxes': fig1_original['boxes']})
        figure2_pose2d_boxes.append({'joints2d': fig2_original['joints2d'], 'boxes': fig2_original['boxes']})

    print("Refining lead and follow track with mid sequence tracking...")
    
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
    valid_fig1_centers.append(__get_pose_center_of_mass(mid_frame['joints3d']))
    valid_fig2_centers.append(__get_pose_center_of_mass(mid_frame2['joints3d']))
    
    # Process forward
    for idx in range(mid_idx + 1, len(figure1_frames)):
        curr_frame1 = figure1_frames[idx]
        curr_frame2 = figure2_frames[idx]
        
        center1 = __get_pose_center_of_mass(curr_frame1['joints3d'])
        center2 = __get_pose_center_of_mass(curr_frame2['joints3d'])
        
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
        
        center1 = __get_pose_center_of_mass(curr_frame1['joints3d'])
        center2 = __get_pose_center_of_mass(curr_frame2['joints3d'])
        
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
    print("Adjusting floor level...")
    figure1_frames, figure2_frames = __adjust_floor_level(figure1_frames, figure2_frames)
    
    # Apply shadow smoothing
    print("Smoothing pose positions on floor with moving average...")
    figure1_frames = __apply_shadow_smoothing(figure1_frames)
    figure2_frames = __apply_shadow_smoothing(figure2_frames)
    
    # Apply joint by joint moving average
    print("Smoothing joints with moving average...")
    figure1_frames = __apply_moving_average(figure1_frames)
    figure2_frames = __apply_moving_average(figure2_frames)

    print("Saving results...")
    # Prepare 3D pose data (xyz format)
    figure1_xyz = [__convert_joints3d_to_xyz_list(pose['joints3d']) for pose in figure1_frames]
    figure2_xyz = [__convert_joints3d_to_xyz_list(pose['joints3d']) for pose in figure2_frames]
    
    # Save separate files
    with open(os.path.join(output_dir, 'figure1.json'), 'w') as f:
        json.dump(figure1_xyz, f, indent=2)
    
    with open(os.path.join(output_dir, 'figure2.json'), 'w') as f:
        json.dump(figure2_xyz, f, indent=2)
        
    with open(os.path.join(output_dir, 'figure1-pose2d-boxes.json'), 'w') as f:
        json.dump(figure1_pose2d_boxes, f, indent=2)
        
    with open(os.path.join(output_dir, 'figure2-pose2d-boxes.json'), 'w') as f:
        json.dump(figure2_pose2d_boxes, f, indent=2)
    
    print(f"Figure tracking data saved to {output_dir}/figure[1,2].json")
    print(f"2D pose and boxes data saved to {output_dir}/figure[1,2]-pose2d-boxes.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adjust 3D poses based on camera movement and track and refine lead and follow dancer.")
    parser.add_argument("--output_dir", help="Path to the output directory", required=True)
    args = parser.parse_args()

    process_poses(args.output_dir)
