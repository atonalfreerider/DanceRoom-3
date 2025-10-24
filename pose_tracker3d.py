import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
from collections import deque
from sklearn.linear_model import RANSACRegressor
from camera_transform import get_camera_model

def __load_vo_data(output_dir: str) -> List[np.ndarray]:
    """Load camera rotation data from Human3r json"""
    poses_path = output_dir + '/poses3d.json'
    
    try:
        with open(poses_path, 'r') as f:
            poses_data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: No poses data found at {poses_path}")
        return []

    # Extract camera rotations from Human3r format
    rotations = []
    frames = poses_data.get('frames', {})
    
    for frame_num in sorted(map(int, frames.keys())):
        frame_str = str(frame_num)
        frame_data = frames[frame_str]
        
        if 'camera' in frame_data and 'rotation' in frame_data['camera']:
            # Camera rotation is already a 3x3 matrix
            rotation_matrix = np.array(frame_data['camera']['rotation'])
            # Invert to get camera-to-world transform
            rotations.append(rotation_matrix.T)
        else:
            # Use identity if no rotation data
            rotations.append(np.eye(3))
    
    return rotations

def __load_poses_data(poses_json_path: str) -> Any | None:
    """Load 3D poses data from Human3r json"""
    
    try:
        with open(poses_json_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: No poses data found at {poses_json_path}")
        return None


def __extract_world_joints(human_data: Dict) -> np.ndarray:
    """
    Extract world joint positions from Human3r format.
    
    Now expects world_joints to be pre-computed by camera_transform.py
    """
    # Check if world_joints dict exists (should be present after camera transform)
    if 'world_joints' in human_data:
        world_joints_dict = human_data['world_joints']
        
        # Extract body joints (should be 22 joints x 3 coords)
        if 'body' in world_joints_dict:
            body_joints = np.array(world_joints_dict['body'])  # Should be (22, 3)
            
            # SMPL has 24 joints, Human3r body has 22
            # We need to add 2 more joints (L_Hand and R_Hand at indices 22, 23)
            if body_joints.shape[0] == 22:
                # Check if we have hand joints to use
                if 'left_hand' in world_joints_dict and 'right_hand' in world_joints_dict:
                    left_hand = np.array(world_joints_dict['left_hand'])
                    right_hand = np.array(world_joints_dict['right_hand'])
                    
                    # Use first joint of each hand as hand position
                    l_hand_pos = left_hand[0] if len(left_hand) > 0 else body_joints[20]  # L_Wrist fallback
                    r_hand_pos = right_hand[0] if len(right_hand) > 0 else body_joints[21]  # R_Wrist fallback
                else:
                    # Fallback: duplicate wrist positions
                    l_hand_pos = body_joints[20]  # L_Wrist
                    r_hand_pos = body_joints[21]  # R_Wrist
                
                # Stack to create 24 joints
                world_joints_24 = np.vstack([body_joints, l_hand_pos, r_hand_pos])
                return world_joints_24
            elif body_joints.shape[0] >= 24:
                return body_joints[:24]
    
    # Error if no world_joints found
    raise ValueError(
        "No world_joints found in human data! "
        "Make sure camera_transform.py has been run to convert camera space to world space."
    )


def __create_placeholder_skeleton(translation: np.ndarray) -> np.ndarray:
    """Create a simple T-pose skeleton as fallback"""
    num_joints = 24
    joints = np.zeros((num_joints, 3))
    
    joints[0] = translation  # Pelvis
    joints[1] = translation + np.array([0.1, 0, 0])    # L_Hip
    joints[2] = translation + np.array([-0.1, 0, 0])   # R_Hip
    joints[3] = translation + np.array([0, 0.1, 0])    # Spine1
    joints[4] = joints[1] + np.array([0, -0.4, 0])     # L_Knee
    joints[5] = joints[2] + np.array([0, -0.4, 0])     # R_Knee
    joints[6] = joints[3] + np.array([0, 0.1, 0])      # Spine2
    joints[7] = joints[4] + np.array([0, -0.4, 0])     # L_Ankle
    joints[8] = joints[5] + np.array([0, -0.4, 0])     # R_Ankle
    joints[9] = joints[6] + np.array([0, 0.1, 0])      # Spine3
    joints[10] = joints[7] + np.array([0, -0.05, 0.1]) # L_Foot
    joints[11] = joints[8] + np.array([0, -0.05, 0.1]) # R_Foot
    joints[12] = joints[9] + np.array([0, 0.1, 0])     # Neck
    joints[13] = joints[9] + np.array([0.05, 0.05, 0]) # L_Collar
    joints[14] = joints[9] + np.array([-0.05, 0.05, 0])# R_Collar
    joints[15] = joints[12] + np.array([0, 0.15, 0])   # Head
    joints[16] = joints[13] + np.array([0.1, 0, 0])    # L_Shoulder
    joints[17] = joints[14] + np.array([-0.1, 0, 0])   # R_Shoulder
    joints[18] = joints[16] + np.array([0.25, 0, 0])   # L_Elbow
    joints[19] = joints[17] + np.array([-0.25, 0, 0])  # R_Elbow
    joints[20] = joints[18] + np.array([0.25, 0, 0])   # L_Wrist
    joints[21] = joints[19] + np.array([-0.25, 0, 0])  # R_Wrist
    joints[22] = joints[20] + np.array([0.1, 0, 0])    # L_Hand
    joints[23] = joints[21] + np.array([-0.1, 0, 0])   # R_Hand
    
    return joints


def __adjust_3d_points(person_data, rotation_matrix: np.ndarray):
    """Transform points from camera space to world space"""
    # Points are already in meters from Human3r world_joints
    points = person_data['joints3d']
    points_array = np.array(points).reshape(-1, 3) / 1000.0  # Convert from mm to m

    # Transform points from camera space to world space
    world_points = np.dot(points_array, rotation_matrix)

    # NOTE: If poses are upside down, we may need to adjust the coordinate system
    # Human3r uses Y-up, Unity uses Y-up, so we should NOT flip Y
    # However, if the camera rotation is inverted, we might need to adjust
    
    # Convert back to millimeters
    person_data['joints3d'] = (world_points * 1000.0).tolist()
    
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


def __convert_joints3d_to_xyz_list(joints3d):
    """Convert joints3d array format to list of xyz dictionaries"""
    return [{'x': joint[0], 'y': joint[1], 'z': joint[2]} for joint in joints3d]

def __calculate_floor_level(all_poses: List[Dict]) -> float:
    """
    Calculate floor level using RANSAC on lowest foot positions.
    
    Args:
        all_poses: List of all pose dictionaries with joints3d
    
    Returns:
        Floor Y position (most negative Y value where feet are)
    """
    lowest_foot_y_values = []
    
    # SMPL joint indices for feet
    LEFT_ANKLE = 7
    RIGHT_ANKLE = 8
    LEFT_FOOT = 10
    RIGHT_FOOT = 11
    
    for pose in all_poses:
        joints = np.array(pose['joints3d']).reshape(-1, 3)
        
        # Get Y values of all foot joints
        foot_y_values = [
            joints[LEFT_ANKLE][1],
            joints[RIGHT_ANKLE][1],
            joints[LEFT_FOOT][1],
            joints[RIGHT_FOOT][1]
        ]
        
        # Take the most negative (lowest) foot position for this frame
        lowest_y = min(foot_y_values)
        lowest_foot_y_values.append(lowest_y)
    
    if not lowest_foot_y_values:
        return 0.0
    
    # Use RANSAC to find robust floor level estimate
    X = np.arange(len(lowest_foot_y_values)).reshape(-1, 1)
    y = np.array(lowest_foot_y_values)
    
    try:
        ransac = RANSACRegressor(random_state=42, min_samples=max(2, len(y) // 10))
        ransac.fit(X, y)
        
        # Get the median of inliers for more stable estimate
        inlier_mask = ransac.inlier_mask_
        floor_level = np.median(y[inlier_mask])
    except:
        # Fallback to median if RANSAC fails
        floor_level = np.median(lowest_foot_y_values)
    
    return float(floor_level)

def __invert_and_translate_poses(poses: List[Dict], floor_y: float) -> List[Dict]:
    """
    Invert XYZ coordinates and translate so floor is at Y=0.
    
    Args:
        poses: List of pose dictionaries
        floor_y: The Y position of the floor (will be translated to 0)
    
    Returns:
        Transformed poses
    """
    for pose in poses:
        joints = np.array(pose['joints3d']).reshape(-1, 3)
        
        # Invert all coordinates (fix mirroring)
        joints = joints * -1.0
        
        # After inversion, floor_y becomes -floor_y
        # To move floor to Y=0, we need to subtract the inverted floor position
        # If original floor was at -1000mm, after inversion it's at +1000mm
        # We need to shift down by 1000mm to get floor at 0
        inverted_floor_y = -floor_y
        joints[:, 1] -= inverted_floor_y
        
        pose['joints3d'] = joints.tolist()
    
    return poses

def process_poses(poses_json_path: str, output_dir: str, camera_model: str = "handheld", fixed_focal_length: bool = True):
    """Process and adjust 3D poses based on camera movement, and track lead and follow dancer"""
    try:
        print("="*60)
        print("POSE TRACKER STARTING")
        print("="*60)
        print(f"Poses JSON: {poses_json_path}")
        print(f"Output directory: {output_dir}")
        print(f"Camera model: {camera_model}")
        print(f"Fixed focal length: {fixed_focal_length}")
        
        print("\nLoading pose data from Human3R output...")
        poses_data = __load_poses_data(poses_json_path)
        
        print(f"Poses data loaded: {poses_data is not None}")
        
        if not poses_data:
            print("\n" + "="*60)
            print("ERROR: Missing required data!")
            print("="*60)
            print(f"- {poses_json_path} could not be loaded")
            return
        
        # Apply camera-to-world transformation
        print(f"\nApplying {camera_model} camera model transformation...")
        cam_model = get_camera_model(camera_model)
        poses_data = cam_model.transform_poses_to_world(poses_data, fixed_focal_length)
        print("✓ Camera transformation complete")
        
        # Get frames from Human3r format
        frames_dict = poses_data.get('frames', {})
        metadata = poses_data.get('metadata', {})
        
        print(f"\nTotal frames in poses JSON: {len(frames_dict)}")
        print(f"Coordinate system: {metadata.get('coordinate_system', 'unknown')}")
        print(f"Camera model: {metadata.get('camera_model', 'unknown')}")
        print(f"Fixed focal length: {metadata.get('fixed_focal_length', 'unknown')}")
        print(f"Units: {metadata.get('units', 'unknown')}")
        
        # Verify world_joints exist
        has_world_joints = False
        
        if frames_dict:
            first_frame_key = next(iter(frames_dict))
            first_frame = frames_dict[first_frame_key]
            if 'humans' in first_frame and len(first_frame['humans']) > 0:
                first_human = first_frame['humans'][0]
                has_world_joints = 'world_joints' in first_human
                
                if has_world_joints:
                    print(f"✓ Found world_joints in actual data")
                    wj = first_human['world_joints']
                    print(f"  world_joints structure: {list(wj.keys())}")
                else:
                    print(f"⚠ No world_joints found!")
                    print(f"  Make sure camera_transform.py was run to convert camera->world space")
        
        if not has_world_joints:
            print("\n" + "="*60)
            print("ERROR: No world_joints found!")
            print("="*60)
            print("The poses3d.json file must have world_joints computed.")
            print("This should be done by camera_transform.py based on the camera model.")
            print("\nExpected workflow:")
            print("1. Human3R generates poses in camera space")
            print("2. camera_transform.py converts to world space")
            print("3. pose_tracker3d.py tracks dancers in world space")
            return
        
        # Convert Human3r format to internal format (now using world_joints)
        frames = {}
        
        for frame_num_str, frame_data in frames_dict.items():
            frame_num = int(frame_num_str)
            humans = frame_data.get('humans', [])
            
            frames[frame_num_str] = []
            for human in humans:
                # Extract world joint positions (already in world space!)
                joints3d = __extract_world_joints(human)
                
                person_data = {
                    'joints3d': joints3d.tolist(),
                    'person_id': human.get('person_id', 0),
                }
                frames[frame_num_str].append(person_data)
        
        print(f"\nProcessed {len(frames)} frames with pose data")
        
        # NOTE: We no longer need to rotate poses since they're already in world space!
        # The camera_transform.py has already applied the necessary transformations
        
        # Lists to store tracked figures
        figure1_frames = []
        figure2_frames = []
        figure1_pose2d_boxes = []
        figure2_pose2d_boxes = []
        previous_figure1_pos = None
        previous_figure1_height = None
        
        print("\nExtracting lead and follow (poses already in world space)...")
        for frame_num in tqdm(sorted(map(int, frames.keys())), desc="Extracting lead and follow"):
            frame_str = str(frame_num)
            frame_data = frames[frame_str]
            
            if not frame_data:
                continue
            
            valid_poses = []
            
            # Process poses (already in world space, no rotation needed)
            for person_data in frame_data:
                adjusted_joints = person_data['joints3d']
                center = __get_pose_center(adjusted_joints)
                height = __calculate_skeletal_height(adjusted_joints)
                distance_to_camera = np.linalg.norm(center)
                
                # Create dummy box data
                dummy_box = [0, 0, 100, 100, 1.0]
                dummy_joints2d = [[0, 0] for _ in range(len(adjusted_joints))]
                
                original_data = {
                    'joints2d': dummy_joints2d,
                    'box': dummy_box
                }
                
                valid_poses.append((person_data, center, height, distance_to_camera, original_data))
            
            # Ensure at least two poses
            if len(valid_poses) < 2:
                continue
            
            # First sort by distance to camera
            valid_poses.sort(key=lambda x: x[3])
            closest_poses = valid_poses[:2]
            
            # Then sort these two by height
            closest_poses.sort(key=lambda x: x[2], reverse=True)
            
            # Tracking logic (unchanged)
            pose1, pos1, height1, _, original_data1 = closest_poses[0]
            pose2, pos2, height2, _, original_data2 = closest_poses[1]
            
            if previous_figure1_pos is None:
                fig1_pose, fig2_pose = pose1, pose2
                fig1_original, fig2_original = original_data1, original_data2
                previous_figure1_pos = pos1
                previous_figure1_height = height1
            else:
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
                    fig1_pose, fig2_pose = pose1, pose2
                    fig1_original, fig2_original = original_data1, original_data2
                    previous_figure1_pos = pos1
                    previous_figure1_height = height1

            figure1_frames.append(fig1_pose)
            figure2_frames.append(fig2_pose)
            figure1_pose2d_boxes.append({'joints2d': fig1_original['joints2d'], 'box': fig1_original['box']})
            figure2_pose2d_boxes.append({'joints2d': fig2_original['joints2d'], 'box': fig2_original['box']})

        print(f"\nCollected {len(figure1_frames)} frames for figure1")
        print(f"Collected {len(figure2_frames)} frames for figure2")

        if len(figure1_frames) == 0 or len(figure2_frames) == 0:
            print("\n" + "="*60)
            print("ERROR: No valid pose pairs found!")
            print("="*60)
            return

        print("\nRefining lead and follow track with mid sequence tracking...")
        
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
        
        frames_since_fig1 = 0
        frames_since_fig2 = 0
        BASE_RADIUS = 500.0  # Use millimeters for radius
        
        mid_frame = figure1_frames[mid_idx]
        mid_frame2 = figure2_frames[mid_idx]
        
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
            
            radius1 = BASE_RADIUS + (frames_since_fig1 * 100)
            radius2 = BASE_RADIUS + (frames_since_fig2 * 100)
            
            dist1 = np.linalg.norm(center1 - valid_fig1_centers[-1])
            dist2 = np.linalg.norm(center2 - valid_fig2_centers[-1])
            
            if dist1 <= radius1:
                valid_fig1_frames.append(curr_frame1)
                valid_fig1_indices.append(idx)
                valid_fig1_centers.append(center1)
                frames_since_fig1 = 0
            else:
                frames_since_fig1 += 1
            
            if dist2 <= radius2:
                valid_fig2_frames.append(curr_frame2)
                valid_fig2_indices.append(idx)
                valid_fig2_centers.append(center2)
                frames_since_fig2 = 0
            else:
                frames_since_fig2 += 1
        
        frames_since_fig1 = 0
        frames_since_fig2 = 0
        
        # Process backward
        for idx in range(mid_idx - 1, -1, -1):
            curr_frame1 = figure1_frames[idx]
            curr_frame2 = figure2_frames[idx]
            
            center1 = __get_pose_center_of_mass(curr_frame1['joints3d'])
            center2 = __get_pose_center_of_mass(curr_frame2['joints3d'])
            
            radius1 = BASE_RADIUS + (frames_since_fig1 * 100)
            radius2 = BASE_RADIUS + (frames_since_fig2 * 100)
            
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

        if not figure1_frames or not figure2_frames:
            print("Error: No valid poses found after processing")
            return

        print("\nSaving results...")
        
        # Combine all poses to calculate floor level
        all_poses = figure1_frames + figure2_frames
        
        print("\nCalculating floor level from foot positions...")
        floor_y = __calculate_floor_level(all_poses)
        print(f"✓ Floor level detected at Y = {floor_y:.3f} mm")
        
        # Invert coordinates and translate to set floor at Y=0
        print("\nInverting coordinates and adjusting floor level...")
        print(f"  After inversion, floor will be at Y = {-floor_y:.3f} mm")
        print(f"  Translating down by {-floor_y:.3f} mm to set floor at Y=0")
        figure1_frames = __invert_and_translate_poses(figure1_frames, floor_y)
        figure2_frames = __invert_and_translate_poses(figure2_frames, floor_y)
        
        figure1_xyz = [__convert_joints3d_to_xyz_list(pose['joints3d']) for pose in figure1_frames]
        figure2_xyz = [__convert_joints3d_to_xyz_list(pose['joints3d']) for pose in figure2_frames]
        
        with open(os.path.join(output_dir, 'figure1.json'), 'w') as f:
            json.dump(figure1_xyz, f, indent=2)
        
        with open(os.path.join(output_dir, 'figure2.json'), 'w') as f:
            json.dump(figure2_xyz, f, indent=2)
            
        with open(os.path.join(output_dir, 'figure1-pose2d-boxes.json'), 'w') as f:
            json.dump(figure1_pose2d_boxes, f, indent=2)
            
        with open(os.path.join(output_dir, 'figure2-pose2d-boxes.json'), 'w') as f:
            json.dump(figure2_pose2d_boxes, f, indent=2)
        
        # Save floor metadata
        floor_metadata = {
            "floor_y_mm": 0.0,  # Floor is now at Y=0
            "original_floor_y_mm": float(floor_y),
            "inverted_floor_y_mm": float(-floor_y),
            "translation_applied_mm": float(-(-floor_y)),  # Amount subtracted from Y
            "coordinates_inverted": True
        }
        
        with open(os.path.join(output_dir, 'floor_metadata.json'), 'w') as f:
            json.dump(floor_metadata, f, indent=2)
        
        print(f"\n✓ Figure tracking data saved to {output_dir}/figure[1,2].json")
        print(f"✓ 2D pose and boxes data saved to {output_dir}/figure[1,2]-pose2d-boxes.json")
        print(f"✓ Floor metadata saved to {output_dir}/floor_metadata.json")
        print("  - Coordinates inverted (fixed mirroring)")
        print(f"  - Floor set to Y=0")
        print("\n" + "="*60)
        print("POSE TRACKER COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("FATAL ERROR IN POSE TRACKER")
        print("="*60)
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        print("="*60)
        raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adjust 3D poses based on camera movement and track and refine lead and follow dancer.")
    parser.add_argument("--poses_json", help="Path to the poses JSON file from Human3R", required=True)
    parser.add_argument("--camera_model", help="Camera model (static, tripod, handheld)", default="handheld")
    parser.add_argument("--fixed_focal_length", help="Fixed focal length", default="True")
    parser.add_argument("--output_dir", help="Path to the output directory", required=True)
    args = parser.parse_args()
    
    # Convert string to boolean
    fixed_focal = args.fixed_focal_length.lower() in ('true', '1', 'yes')

    process_poses(args.poses_json, args.output_dir, args.camera_model, fixed_focal)
