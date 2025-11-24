 # Copyright 2024 the authors of NeuRAD and contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for Simulator dataset"""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple, Type
import pandas as pd
from pypcd4 import PointCloud
import numpy as np
import pyquaternion
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import numpy.typing as npt
try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required for reading PCD files. Install it with: pip install open3d")

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras.lidars import Lidars, LidarType
from nerfstudio.data.dataparsers.ad_dataparser import (
    DUMMY_DISTANCE_VALUE,
    OPENCV_TO_NERFSTUDIO,
    ADDataParser,
    ADDataParserConfig,
)
from nerfstudio.utils import poses as pose_utils

# Default lidar parameters
HORIZONTAL_BEAM_DIVERGENCE = 3.0e-3 
VERTICAL_BEAM_DIVERGENCE = 1.5e-3
LIDAR_ROTATION_TIME = 0.1
# Maximum reflectance value for normalization
MAX_REFLECTANCE_VALUE = 255.0
EGO_OBJECT_ID = "10010"
STATIONARY_DISPLACEMENT_THRESHOLD = 0.25  # meters

# Allowed classes for trajectory loading
ALLOWED_RIGID_CLASSES = {
    "TYPE_VEHICLE",
    "TYPE_SMALL_CAR",
    "TYPE_MEDIUM_CAR",
    "TYPE_COMPACT_CAR",
    "TYPE_LUXURY_CAR",
    "TYPE_HEAVY_TRUCK",
    "TYPE_BUS",
    "TYPE_SEMITRACTOR",
    "TYPE_SEMITRAILER",
    "TYPE_MOTORBIKE",
    "TYPE_DUMMY_BICYCLE",
}

ALLOWED_DEFORMABLE_CLASSES = {
    "TYPE_PEDESTRIAN",
    "TYPE_DUMMY_CYCLIST",
    "TYPE_MOTORCYCLE_RIDER",
    "TYPE_UNKNOWN",  # Might be pedestrian
}

SIMULATOR_TO_OPENCV = np.array(
    [
        [0, 0, 1],
        [-1, 0, 0],
        [0, -1, 0],
    ]
)
LANE_SHIFT_SIGN: Dict[str, Literal[-1, 1]] = defaultdict(lambda: -1)
LANE_SHIFT_SIGN.update(
    {
        "001": 1,
    }
)


@dataclass
class SimulatorDataParserConfig(ADDataParserConfig):
    """Simulator dataset config.
    
    Custom dataparser for simulator datasets with the following structure:
    - point_clouds/agent_X/lidar_Y/ containing PCD files
    - images/agent_X/camera_Y/ containing PNG images
    - ground_truth/agent_X/camera_Y/ containing annotation files
    - labels/sensor_parameters.json with sensor intrinsics/extrinsics
    - labels/label_nerf_sample.json with OpenLabel annotations
    """

    _target: Type = field(default_factory=lambda: SimulatorDataParser)
    """target class to instantiate"""
    data: Path = Path("/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/Simulator_1Nov2025")
    """Directory specifying location of data."""
    """Name of the sequence/scene to load."""
    start_frame: int = 1
    """Start frame"""
    end_frame: Optional[int] = None
    """End frame"""
    agent_id: str = "agent_1"
    """ID of the agent to load."""
    
    cameras: Tuple[Literal["camera_1", "camera_2", "camera_3", "camera_4", "all"], ...] = (
        "camera_1",
        "camera_2",
        "camera_3",
        "camera_4",
    )
    """Which cameras to use. Can specify individual cameras or 'all' for all available. At least one camera is required."""
    lidars: Tuple[Literal["lidar_1"], ...] = ("lidar_1",)
    """Which lidars to use. Must specify at least one lidar."""
    annotation_interval: float = 0.1
    """The time interval at which the sequence is annotated (s)."""
    rolling_shutter_time: float = 0.0
    """The rolling shutter time for the cameras (seconds)."""
    time_to_center_pixel: float = 0.0
    """The time offset for the center pixel, relative to the image timestamp (seconds)."""
    stationary_displacement_threshold: float = STATIONARY_DISPLACEMENT_THRESHOLD
    """Minimum translation (m) for an actor to be considered dynamic."""
    # distortion_params: Optional[List[float]]  = field(
    #     default_factory=lambda: [
    #         0.00343299, -0.01468419,  0.24592364, -0.00815786
    #     ]
    # )
    """The distortion parameters for the cameras."""
    def __post_init__(self):
        """Validate that at least one camera and one lidar are specified."""
        # Don't call parent __post_init__ as it converts "none" to empty tuples
        # Instead, validate that we have required sensors
        if not self.cameras or len(self.cameras) == 0:
            raise ValueError("At least one camera must be specified. Cannot use empty cameras list.")
        
        if "none" in self.cameras:
            raise ValueError("Cannot use 'none' for cameras. At least one camera is required.")
        
        if not self.lidars or len(self.lidars) == 0:
            raise ValueError("At least one lidar must be specified. Cannot use empty lidars list.")
        
        if "none" in self.lidars:
            raise ValueError("Cannot use 'none' for lidars. At least one lidar is required.")
        
        # Call parent assertions (but skip the "none" conversion)
        assert self.cameras or self.lidars, "Must specify at least one sensor to load"
        assert self.annotation_interval > 1e-6, "Child classes must specify the annotation interval"
        assert self.dataset_start_fraction >= 0.0, "Dataset start fraction must be >= 0.0"
        assert self.dataset_end_fraction <= 1.0, "Dataset end fraction must be <= 1.0"
        assert self.dataset_start_fraction < self.dataset_end_fraction, "Dataset start must be < dataset end"


@dataclass
class SimulatorDataParser(ADDataParser):
    """Simulator DatasetParser"""

    config: SimulatorDataParserConfig

    def __init__(self, config: SimulatorDataParserConfig):
        super().__init__(config)
        self.data_dir = Path(config.data)
        self.agent_id = config.agent_id
        self.label_file = self.data_dir / "labels" / "label_nerf_sample.json"
        self.sensor_params_file = self.data_dir / "labels" / "sensor_parameters.json"
        self.ego_object_id = EGO_OBJECT_ID
        self.stationary_displacement_threshold = config.stationary_displacement_threshold
        
        # Load sensor parameters first (needed for "all" cameras check)
        if not self.sensor_params_file.exists():
            raise ValueError(f"Sensor parameters file not found at {self.sensor_params_file}")
        with open(self.sensor_params_file, "r") as f:
            self.sensor_params = json.load(f)
        
        # Handle "all" for cameras
        if "all" in config.cameras:
            # Get all available cameras from sensor params
            agent_sensors = self.sensor_params.get(self.agent_id, {})
            available_cameras = [k for k in agent_sensors.keys() if k.startswith("camera_")]
            if not available_cameras:
                raise ValueError("No cameras available. Check sensor_parameters.json or camera configuration.")
            self.config.cameras = tuple(available_cameras)
        
        # Load label file
        if not self.label_file.exists():
            raise ValueError(f"Label file not found at {self.label_file}")
        with open(self.label_file, "r") as f:
            self.label_data = json.load(f)
        
        # Parse frames from label file
        self.frames_data = self.label_data.get("openlabel", {}).get("frames", {})
        self.streams_data = self.label_data.get("openlabel", {}).get("streams", {})
        
        # Pre-compute ego poses cache for all frames
        self._ego_poses_cache: Dict[float, np.ndarray] = {}
        # Cache for ground truth files
        self._ego_files_cache: Dict[Tuple[str, float], Dict] = {}
        self._sensor_files_cache: Dict[Tuple[str, float], Dict] = {}
        # self._compute_ego_poses_cache()
        # output_path = Path("ego_poses.json")

    

    def _load_ego_file(self, camera_id: str, timestamp: float) -> Optional[Dict]:
        """Load ego file with caching."""
        cache_key = (camera_id, timestamp)
        if cache_key in self._ego_files_cache and self._ego_files_cache[cache_key] is not None:
            return self._ego_files_cache[cache_key]
        
        sec = int(timestamp)
        msec = int((timestamp - sec) * 10) 
        
        gt_dir = self.data_dir / "ground_truth" / self.agent_id / camera_id
        ego_file = gt_dir / f"ego_{sec}.{msec}00000.json"
        
        if not ego_file.exists():
            self._ego_files_cache[cache_key] = None
            return None
        
        try:
            with open(ego_file, "r") as f:
                ego_data = json.load(f)
            self._ego_files_cache[cache_key] = ego_data
            return ego_data
        except Exception:
            self._ego_files_cache[cache_key] = None
            return None
    def export_ego_poses_to_json(self, output_path: Path) -> None:
        """Export all cached ego poses to JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        poses_list = []
        
        # Sort by timestamp to ensure consistent ordering
        for timestamp in sorted(self._ego_poses_cache.keys()):
            pose = self._ego_poses_cache[timestamp]
            
            # Extract position (translation)
            position = {
                "x": float(pose[0, 3]),
                "y": float(pose[1, 3]),
                "z": float(pose[2, 3])
            }
            
            # Convert rotation matrix to quaternion
            rot_matrix = pose[:3, :3]
            quat = pyquaternion.Quaternion(matrix=rot_matrix)
            
            # Format quaternion as w, x, y, z
            heading = {
                "w": float(quat.w),
                "x": float(quat.x),
                "y": float(quat.y),
                "z": float(quat.z)
            }
            
            poses_list.append({
                "position": position,
                "heading": heading
            })
        
        # Write to JSON file
        with open(output_path, "w") as f:
            json.dump(poses_list, f, indent=2)
        
        print(f"Exported {len(poses_list)} ego poses to {output_path}")


    def _get_object_pose_from_ego_file(self, camera_id: str, timestamp: float, obj_id: str) -> Optional[np.ndarray]:
        """Get object pose in ego frame from ego_*.json file."""
        ego_data = self._load_ego_file(camera_id, timestamp)
        if ego_data is None:
            return None
        
        try:
            moving_objects = ego_data.get("moving_object", [])
            for obj in moving_objects:
                if obj.get("id", {}).get("value") == obj_id:
                    bboxcoord = obj.get("bboxcoord", {})
                    position = bboxcoord.get("position", {})
                    orientation = bboxcoord.get("orientation", {})
                    
                    x = position.get("x", 0.0)
                    y = position.get("y", 0.0)
                    z = position.get("z", 0.0)
                    roll = orientation.get("roll", 0.0)
                    pitch = orientation.get("pitch", 0.0)
                    yaw = orientation.get("yaw", 0.0)
                    
                    rot = _euler_to_rotation_matrix(roll, pitch, yaw)
                    pose = np.eye(4)
                    pose[:3, :3] = rot
                    pose[:3, 3] = [x, y, z]
                    return pose
        except Exception:
            pass
        
        return None

    def _load_sensor_file(self, camera_id: str, timestamp: float) -> Optional[Dict]:
        """Load sensor file with caching."""
        cache_key = (camera_id, timestamp)
        if cache_key in self._sensor_files_cache:
            return self._sensor_files_cache[cache_key]
        
        sec = int(timestamp)
        msec = int((timestamp - sec) * 1000)
        
        gt_dir = self.data_dir / "ground_truth" / self.agent_id / camera_id
        sensor_file = gt_dir / f"sensor_{sec}.{msec}.json"
        
        if not sensor_file.exists():
            self._sensor_files_cache[cache_key] = None
            return None
        
        try:
            with open(sensor_file, "r") as f:
                sensor_data = json.load(f)
            self._sensor_files_cache[cache_key] = sensor_data
            return sensor_data
        except Exception:
            self._sensor_files_cache[cache_key] = None
            return None

    def _get_object_pose_from_sensor_file(self, camera_id: str, timestamp: float, obj_id: str) -> Optional[np.ndarray]:
        """Get object pose in sensor frame from sensor_*.json file."""
        sensor_data = self._load_sensor_file(camera_id, timestamp)
        if sensor_data is None:
            return None
        
        try:
            moving_objects = sensor_data.get("moving_object", [])
            for obj in moving_objects:
                if obj.get("id", {}).get("value") == obj_id:
                    bboxcoord = obj.get("bboxcoord", {})
                    position = bboxcoord.get("position", {})
                    orientation = bboxcoord.get("orientation", {})
                    
                    x = position.get("x", 0.0)
                    y = position.get("y", 0.0)
                    z = position.get("z", 0.0)
                    roll = orientation.get("roll", 0.0)
                    pitch = orientation.get("pitch", 0.0)
                    yaw = orientation.get("yaw", 0.0)
                    
                    rot = _euler_to_rotation_matrix(roll, pitch, yaw)
                    pose = np.eye(4)
                    pose[:3, :3] = rot
                    pose[:3, 3] = [x, y, z]
                    return pose
        except Exception:
            pass
        
        return None

    def _get_frame_data_by_timestamp(self, timestamp: float) -> Optional[Dict]:
        """Get frame data for a given timestamp (assumes exact match)."""
        for frame_id, frame_data in self.frames_data.items():
            frame_props = frame_data.get("frame_properties", {})
            frame_ts = frame_props.get("timestamp", None)
            if frame_ts is not None and abs(frame_ts - timestamp) < 1e-6:  # Exact match
                return frame_data
        return None

    def _compute_ego_pose_from_object_poses(
        self, timestamp: float, camera_name: str
    ) -> Optional[np.ndarray]:
        """Compute ego pose in world frame using object poses.
        
        Uses: E_object_in_world = E_ego_in_world @ E_object_in_ego
        So: E_ego_in_world = E_object_in_world @ inv(E_object_in_ego)
        
        Uses multiple objects to get robust estimate.
        """
        # Get frame data for this timestamp (assumes exact match)
        frame_data = self._get_frame_data_by_timestamp(timestamp)
        if frame_data is None:
            return None
        
        objects = frame_data.get("objects", {})
        
        # Collect valid object poses for this camera
        ego_pose_candidates = []
        
        for obj_id, obj_data in objects.items():
            # Get object pose in world frame from label file
            cuboid = obj_data.get("object_data", {}).get("cuboid", {})
            
            cuboid_value = cuboid.get("value", [])
            E_object_in_world, _ = cuboid_to_pose_and_dims(cuboid_value)
            
            # Get object pose in ego frame
            E_object_in_ego = self._get_object_pose_from_ego_file(camera_name, timestamp, obj_id)
            
            # Compute ego pose: E_ego_in_world = E_object_in_world @ inv(E_object_in_ego)
            E_ego_in_world = E_object_in_world @ np.linalg.inv(E_object_in_ego)
            ego_pose_candidates.append(E_ego_in_world)
        
        if len(ego_pose_candidates) == 1:
            return ego_pose_candidates[0]
        
        # Sanity check: verify all candidates are consistent
        # Compare all pairs to check if they're the same (within tolerance)
        translation_tolerance = 0.01  # 1cm tolerance
        rotation_tolerance = 0.001  # ~0.057 degrees tolerance
        
        reference_pose = ego_pose_candidates[0]
        all_consistent = True
        
        for candidate_pose in ego_pose_candidates[1:]:
            # Check translation difference
            trans_error = np.linalg.norm(reference_pose[:3, 3] - candidate_pose[:3, 3])
            if trans_error > translation_tolerance:
                all_consistent = False
                break
            
            # Check rotation difference (angle between rotations)
            R1, R2 = reference_pose[:3, :3], candidate_pose[:3, :3]
            R_diff = R1.T @ R2
            trace = np.trace(R_diff)
            angle_error = np.arccos(np.clip((trace - 1) / 2, -1, 1))
            if angle_error > rotation_tolerance:
                all_consistent = False
                break
        print(f"All consistent: {all_consistent}, timestamp: {timestamp}, camera_name: {camera_name}")
        # If all candidates are consistent, return the reference pose
        # Otherwise, still return it but it indicates there might be an issue
        # (The sanity check function will catch this)
        return reference_pose

    def _compute_ego_poses_cache(self):
        """Pre-compute and cache ego poses for all timestamps."""
        # Collect all unique timestamps
        timestamps = set()
        for frame_id, frame_data in self.frames_data.items():
            frame_props = frame_data.get("frame_properties", {})
            timestamp = frame_props.get("timestamp", None)
            if timestamp is not None:
                timestamps.add(timestamp)
        
        # For each timestamp, compute ego pose using first available camera
        for timestamp in sorted(timestamps):
            # Try each camera until we find one with valid data
            ego_pose = None
            for camera_name in self.config.cameras:
                ego_pose = self._compute_ego_pose_from_object_poses(timestamp, camera_name)
                if ego_pose is not None:
                    break
            
            if ego_pose is not None:
                self._ego_poses_cache[timestamp] = ego_pose

    def _get_ego_pose(self, timestamp: float) -> np.ndarray:
        """Get ego pose for given timestamp (assumes exact match)."""
        if timestamp in self._ego_poses_cache:
            return self._ego_poses_cache[timestamp]
        
        # Try to find exact match (accounting for floating point precision)
        for cached_ts in self._ego_poses_cache.keys():
            if abs(cached_ts - timestamp) < 1e-6:
                return self._ego_poses_cache[cached_ts]
        
        raise ValueError(f"No ego pose available for timestamp {timestamp}")

    def _get_lane_shift_sign(self, sequence: str) -> Literal[-1, 1]:
        return LANE_SHIFT_SIGN.get(sequence, 1)
    
    def _get_cameras(self) -> Tuple[Cameras, List[Path]]:
        """Returns camera info and image filenames."""
        poses = []
        intrinsics = []
        heights = []
        widths = []
        times = []
        filenames = []
        distortion_params = []
        idxs = []
        cameras = list(self.config.cameras)
        if self.config.end_frame is None:
            end_frame = len(self.frames_data)
        else:
            end_frame = self.config.end_frame
        
        for frame_id in self.frames_data.keys():
            if int(frame_id) < self.config.start_frame or int(frame_id) >= end_frame:
                continue
            
            frame_data = self.frames_data[frame_id]
            for camera_id in self.config.cameras:
                # ego_pose_in_world = self._ego_poses_cache[self.frames_data[frame_id]["frame_properties"]["timestamp"]]


                ego_pose_in_world, _ = cuboid_to_pose_and_dims(
                    self.frames_data[frame_id]["objects"][self.ego_object_id]["object_data"]["cuboid"]["value"]
                )
                sensor_in_ego_dict = self.sensor_params[f"{self.config.agent_id}"][f"{camera_id}"]["extrinsic"]
                sensor_in_ego = _extrinsic_to_matrix(sensor_in_ego_dict)
                Rotation = sensor_in_ego[:3, :3]
                Rotation = Rotation @ SIMULATOR_TO_OPENCV
                sensor_in_ego[:3, :3] = Rotation
                pose = ego_pose_in_world @ sensor_in_ego 
                pose[:3, :3] = pose[:3, :3] @ OPENCV_TO_NERFSTUDIO
                poses.append(pose[:3, :4])
                intrinsic = self.sensor_params[f"{self.config.agent_id}"][f"{camera_id}"]["intrinsic"]
                fx = intrinsic.get("fx", 600.0)
                fy = intrinsic.get("fy", 600.0)
                cx = intrinsic.get("cx", 600.0)
                cy = intrinsic.get("cy", 388.5)
                intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                intrinsics.append(intrinsic_matrix)
                filenames.append( self.config.data / self.frames_data[frame_id]["frame_properties"]["streams"][f"{self.config.agent_id}/{camera_id}"]["uri"])
                times.append(self.frames_data[frame_id]["frame_properties"]["timestamp"])
                idxs.append(cameras.index(camera_id))
                heights.append(self.streams_data[f"{self.config.agent_id}/{camera_id}"]["stream_properties"]["height"] - (250 if camera_id == "camera_1" else 0))
                widths.append(self.streams_data[f"{self.config.agent_id}/{camera_id}"]["stream_properties"]["width"])
                distortion_params.append([-0.10086563, 0.06112929, -0.04727966, 0.00974163, 0.0, 0.0])
        # Convert to tensors
        intrinsics = torch.from_numpy(np.array(intrinsics)).float()
        poses = torch.tensor(np.array(poses), dtype=torch.float32)
        times = torch.tensor(times, dtype=torch.float64)
        idxs = torch.tensor(idxs).int().unsqueeze(-1)
        heights = torch.tensor(heights).int()
        widths = torch.tensor(widths).int()
        distortion_params = torch.tensor(np.array(distortion_params), dtype=torch.float32)
        cameras = Cameras(
            fx=intrinsics[:, 0, 0],
            fy=intrinsics[:, 1, 1],
            cx=intrinsics[:, 0, 2],
            cy=intrinsics[:, 1, 2],
            height=heights,
            width=widths,
            distortion_params=distortion_params,
            camera_to_worlds=poses,
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            metadata={"sensor_idxs": idxs},
        )
        return cameras, filenames

    
    def _get_lidars(self) -> Tuple[Lidars, List[Path]]:
        """Returns lidar info and filenames."""
        poses = []
        times = []
        filenames = []
        idxs = []

        if self.config.end_frame is None:
            end_frame = len(self.frames_data)
        else:
            end_frame = self.config.end_frame
        
        for frame_id in self.frames_data.keys():
            if int(frame_id) < self.config.start_frame or int(frame_id) >= end_frame:
                continue

            for lidar_id in self.config.lidars:
                # ego_pose_in_world = self._ego_poses_cache[self.frames_data[frame_id]["frame_properties"]["timestamp"]]
                ego_pose_in_world, _ = cuboid_to_pose_and_dims(
                    self.frames_data[frame_id]["objects"][self.ego_object_id]["object_data"]["cuboid"]["value"]
                )
                sensor_in_ego_dict = self.sensor_params[f"{self.config.agent_id}"][f"{lidar_id}"]["extrinsic"]
                sensor_in_ego = _extrinsic_to_matrix(sensor_in_ego_dict)
                # Lidar uses same coordinate convention as world frame (X forward, Y left, Z up)
                # No SIMULATOR_TO_OPENCV transformation needed for lidar
                pose = ego_pose_in_world @ sensor_in_ego
                poses.append(torch.from_numpy(pose[:3, :4]).float())
                times.append(self.frames_data[frame_id]["frame_properties"]["timestamp"])
                filenames.append(self.config.data / self.frames_data[frame_id]["frame_properties"]["streams"][f"{self.config.agent_id}/{lidar_id}"]["uri"].replace(".0", ""))
                idxs.append(list(self.config.lidars).index(lidar_id))
        
        
        poses = torch.stack(poses)
        times = torch.tensor(times, dtype=torch.float64)  # need higher precision
        idxs = torch.tensor(idxs).int().unsqueeze(-1)

        lidars = Lidars(
            lidar_to_worlds=poses[:, :3, :4],
            lidar_type=LidarType.VELODYNE16,
            times=times,
            metadata={"sensor_idxs": idxs},
            horizontal_beam_divergence=HORIZONTAL_BEAM_DIVERGENCE,
            vertical_beam_divergence=VERTICAL_BEAM_DIVERGENCE,
            valid_lidar_distance_threshold=DUMMY_DISTANCE_VALUE / 2,
        )
        
        return lidars, filenames


    def _read_lidars(self, lidars: Lidars, filenames: List[Path]) -> List[torch.Tensor]:
        """Reads the point clouds from the given filenames."""
        point_clouds = []

        for i, filepath in enumerate(filenames):
            lidar = lidars[i]
            pcd_data = PointCloud.from_path(str(filepath)).pc_data
            l2w = pose_utils.to4x4(lidar.lidar_to_worlds)
            points = pd.DataFrame(pcd_data).to_numpy()
            xyz = points[:, :3]  # N x 3
            intensity = points[:, 3] # N x 1 
            intensity /= MAX_REFLECTANCE_VALUE  # N,
            t = get_mock_timestamps(xyz)  # N, relative timestamps
            pc = np.hstack((xyz, intensity[:, None], t[:, None]))
            point_clouds.append(torch.from_numpy(pc).float())

        lidars.lidar_to_worlds = lidars.lidar_to_worlds.float()
        return point_clouds


    def _get_actor_trajectories(self) -> List[Dict]:
        """Returns a list of actor trajectories from label file."""
        trajectories = []
        
        # Group objects by ID across frames
        object_trajectories: Dict[str, List[Tuple[float, np.ndarray, np.ndarray, str]]] = defaultdict(list)
        
        for frame_id, frame_data in self.frames_data.items():
            frame_props = frame_data.get("frame_properties", {})
            timestamp = frame_props.get("timestamp", None)
            
            
            objects = frame_data.get("objects", {})
            
            for obj_id, obj_data in objects.items():
                obj_type = obj_data.get("object_data", {}).get("type", "")
                cuboid = obj_data.get("object_data", {}).get("cuboid", {})
                cuboid_value = cuboid.get("value", [])
            
                pose, dims = cuboid_to_pose_and_dims(cuboid_value)
                object_trajectories[obj_id].append((timestamp, pose, dims, obj_type))
        
        # Convert to trajectory format
        for obj_id, traj_data in object_trajectories.items():
            if len(traj_data) < 2:
                continue  # Need at least 2 timestamps for trajectory
            
            # Sort by timestamp
            traj_data.sort(key=lambda x: x[0])
            
            timestamps = torch.tensor([t[0] for t in traj_data], dtype=torch.float64)
            pose_list = [t[1] for t in traj_data]
            poses = torch.tensor(pose_list, dtype=torch.float32)
            dims_list = [t[2] for t in traj_data]
            obj_type = traj_data[0][3]
            
            # Use average dimensions
            dims = np.mean(dims_list, axis=0)
            dims = torch.tensor(dims, dtype=torch.float32)
            
            # Determine properties
            positions = np.array([pose[:3, 3] for pose in pose_list])
            displacement = np.linalg.norm(positions[-1] - positions[0])
            is_stationary = displacement < self.stationary_displacement_threshold
            is_rigid = obj_type in ALLOWED_RIGID_CLASSES
            is_deformable = obj_type in ALLOWED_DEFORMABLE_CLASSES
            is_symmetric = obj_type in {
                "TYPE_VEHICLE",
                "TYPE_SMALL_CAR",
                "TYPE_MEDIUM_CAR",
                "TYPE_COMPACT_CAR",
                "TYPE_LUXURY_CAR",
            }
            
            trajectories.append(
                {
                    "poses": poses,
                    "timestamps": timestamps,
                    "dims": dims,
                    "label": obj_type,
                    "stationary": is_stationary,
                    "symmetric": is_symmetric,
                    "deformable": is_deformable,
                    "is_ego": obj_id == self.ego_object_id,
                    "object_id": obj_id,
                }
            )
        
        return trajectories



def cuboid_to_pose_and_dims(cuboid_value: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert cuboid value to pose and dimensions.
    
    Cuboid format: [x, y, z, qx, qy, qz, qw, length, width, height]
    Returns: (4x4 pose matrix, [width, length, height] dimensions)
    """
    x, y, z = cuboid_value[0], cuboid_value[1], cuboid_value[2]
    qx, qy, qz, qw = cuboid_value[3], cuboid_value[4], cuboid_value[5], cuboid_value[6]
    length, width, height = cuboid_value[7], cuboid_value[8], cuboid_value[9]
    
    # Convert quaternion (x, y, z, w) to pyquaternion format (w, x, y, z)
    quat = pyquaternion.Quaternion(w=qw, x=qx, y=qy, z=qz)
    rot_matrix = quat.rotation_matrix
    
    pose = np.eye(4)
    pose[:3, :3] = rot_matrix
    pose[:3, 3] = [x, y, z]
    
    # Dimensions in wlh order (width, length, height)
    dims = np.array([length, width, height], dtype=np.float32)
    
    return pose, dims

def _euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (roll, pitch, yaw) to rotation matrix"""
    r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    return r.as_matrix()

def _extrinsic_to_matrix(extrinsic: Dict) -> np.ndarray:
    """Convert extrinsic parameters (x, y, z, roll, pitch, yaw) to 4x4 transformation matrix"""
    rel = extrinsic.get("relative", {})
    x = rel.get("x", 0.0)
    y = rel.get("y", 0.0)
    z = rel.get("z", 0.0)
    roll = rel.get("roll", 0.0)
    pitch = rel.get("pitch", 0.0)
    yaw = rel.get("yaw", 0.0)
    
    rot = _euler_to_rotation_matrix(roll, pitch, yaw)
    pose = np.eye(4)
    pose[:3, :3] = rot
    pose[:3, 3] = [x, y, z]
    return pose

def get_mock_timestamps(points: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Get mock relative timestamps for the velodyne points."""
    # the velodyne has x forward, y left, z up and the sweep is split behind the car.
    # it is also rotating counter-clockwise, meaning that the angles close to -pi are the
    # first ones in the sweep and the ones close to pi are the last ones in the sweep.
    angles = np.arctan2(points[:, 1], points[:, 0])  # N, [-pi, pi]
    angles += np.pi  # N, [0, 2pi]
    # see how much of the rotation have finished
    fraction_of_rotation = angles / (2 * np.pi)  # N, [0, 1]
    # get the pseudo timestamps based on the total rotation time
    timestamps = fraction_of_rotation * LIDAR_ROTATION_TIME
    return timestamps
