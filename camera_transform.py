"""
Camera Transformation Utilities

Handles conversion from camera space to world space based on camera model.
"""

import numpy as np
import json
from typing import Dict, List, Tuple


class CameraModel:
    """Base class for camera models"""
    
    def __init__(self, name: str):
        self.name = name
    
    def transform_poses_to_world(
        self, 
        poses_data: Dict,
        fixed_focal_length: bool = True
    ) -> Dict:
        """
        Transform poses from camera space to world space.
        
        Args:
            poses_data: Poses data from Human3R JSON
            fixed_focal_length: Whether focal length is fixed
            
        Returns:
            Transformed poses data with world_joints added
        """
        raise NotImplementedError


class StaticCameraModel(CameraModel):
    """
    Static Camera: Camera does not move at all.
    Camera space = World space (identity transform)
    """
    
    def __init__(self):
        super().__init__("static")
    
    def transform_poses_to_world(self, poses_data: Dict, fixed_focal_length: bool = True) -> Dict:
        """
        For static camera, camera joints ARE world joints (identity transform).
        """
        frames = poses_data.get('frames', {})
        
        for frame_key, frame_data in frames.items():
            for human in frame_data.get('humans', []):
                # Copy camera_joints to world_joints (they're the same for static camera)
                if 'camera_joints' in human:
                    human['world_joints'] = human['camera_joints'].copy()
        
        # Add metadata
        poses_data['metadata']['camera_model'] = self.name
        poses_data['metadata']['fixed_focal_length'] = fixed_focal_length
        
        return poses_data


class TripodCameraModel(CameraModel):
    """
    Tripod Camera: Camera can rotate but position is fixed.
    Only apply rotation to transform to world space.
    """
    
    def __init__(self):
        super().__init__("tripod")
    
    def transform_poses_to_world(self, poses_data: Dict, fixed_focal_length: bool = True) -> Dict:
        """
        Apply only camera rotation to get world coordinates.
        Camera position is assumed to be at origin (0,0,0).
        """
        frames = poses_data.get('frames', {})
        
        for frame_key, frame_data in frames.items():
            camera = frame_data.get('camera', {})
            R = np.array(camera.get('rotation_matrix', np.eye(3)))  # 3x3 camera-to-world rotation
            
            for human in frame_data.get('humans', []):
                if 'camera_joints' not in human:
                    continue
                
                world_joints = {}
                camera_joints = human['camera_joints']
                
                # Transform each joint group
                for joint_type in ['body', 'face', 'left_hand', 'right_hand']:
                    if joint_type in camera_joints:
                        joints_cam = np.array(camera_joints[joint_type])  # Shape: (N, 3)
                        
                        # Apply rotation only (no translation for tripod)
                        joints_world = (R @ joints_cam.T).T  # Shape: (N, 3)
                        
                        world_joints[joint_type] = joints_world.tolist()
                
                human['world_joints'] = world_joints
        
        # Add metadata
        poses_data['metadata']['camera_model'] = self.name
        poses_data['metadata']['fixed_focal_length'] = fixed_focal_length
        
        return poses_data


class HandheldCameraModel(CameraModel):
    """
    Handheld Camera: Camera can both translate and rotate.
    Apply full camera-to-world transformation (rotation + translation).
    """
    
    def __init__(self):
        super().__init__("handheld")
    
    def transform_poses_to_world(self, poses_data: Dict, fixed_focal_length: bool = True) -> Dict:
        """
        Apply full camera transformation (rotation + translation).
        """
        frames = poses_data.get('frames', {})
        
        for frame_key, frame_data in frames.items():
            camera = frame_data.get('camera', {})
            R = np.array(camera.get('rotation_matrix', np.eye(3)))  # 3x3 rotation
            t = np.array(camera.get('translation', [0, 0, 0]))  # 3D translation
            
            for human in frame_data.get('humans', []):
                if 'camera_joints' not in human:
                    continue
                
                world_joints = {}
                camera_joints = human['camera_joints']
                
                # Transform each joint group
                for joint_type in ['body', 'face', 'left_hand', 'right_hand']:
                    if joint_type in camera_joints:
                        joints_cam = np.array(camera_joints[joint_type])  # Shape: (N, 3)
                        
                        # Apply full transformation: p_world = R @ p_cam + t
                        joints_world = (R @ joints_cam.T).T + t  # Shape: (N, 3)
                        
                        world_joints[joint_type] = joints_world.tolist()
                
                human['world_joints'] = world_joints
        
        # Add metadata
        poses_data['metadata']['camera_model'] = self.name
        poses_data['metadata']['fixed_focal_length'] = fixed_focal_length
        
        return poses_data


def get_camera_model(model_name: str) -> CameraModel:
    """
    Factory function to get appropriate camera model.
    
    Args:
        model_name: One of 'static', 'tripod', 'handheld'
    
    Returns:
        CameraModel instance
    """
    models = {
        'static': StaticCameraModel,
        'tripod': TripodCameraModel,
        'handheld': HandheldCameraModel
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown camera model: {model_name}. Must be one of {list(models.keys())}")
    
    return models[model_name]()


def apply_camera_transform(
    input_json_path: str,
    output_json_path: str,
    camera_model: str,
    fixed_focal_length: bool = True
) -> None:
    """
    Apply camera-to-world transformation to poses JSON.
    
    Args:
        input_json_path: Path to Human3R poses JSON (camera space)
        output_json_path: Path to save transformed JSON (world space)
        camera_model: Camera model name ('static', 'tripod', 'handheld')
        fixed_focal_length: Whether focal length is fixed
    """
    # Load poses data
    with open(input_json_path, 'r') as f:
        poses_data = json.load(f)
    
    # Get appropriate camera model
    cam_model = get_camera_model(camera_model)
    
    # Transform poses
    transformed_data = cam_model.transform_poses_to_world(poses_data, fixed_focal_length)
    
    # Update metadata
    metadata = transformed_data.get('metadata', {})
    metadata['coordinate_system'] = 'world'
    metadata['source_coordinate_system'] = 'camera'
    metadata['transformation_applied'] = True
    
    # Save transformed data
    with open(output_json_path, 'w') as f:
        json.dump(transformed_data, f, indent=2)
    
    print(f"✓ Applied {camera_model} camera model transformation")
    print(f"  Input:  {input_json_path}")
    print(f"  Output: {output_json_path}")
