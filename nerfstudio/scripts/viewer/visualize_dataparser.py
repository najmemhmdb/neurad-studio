#!/usr/bin/env python
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

"""
Visualization utility for SimulatorDataParser.

This script visualizes camera poses, lidar poses, and point clouds from the 
SimulatorDataParser without running any NeRF training. Useful for debugging 
and verifying that the data is loaded correctly.

Example usage:
    python nerfstudio/scripts/viewer/visualize_dataparser.py \
        --data /path/to/simulator/data \
        --agent-id agent_1 \
        --start-frame 1 \
        --end-frame 10
"""

from __future__ import annotations

import time
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import tyro
import viser
import viser.transforms as vtf
from PIL import Image, ImageDraw, ImageFont

from nerfstudio.cameras.lidars import intensity_to_rgb, transform_points
from nerfstudio.data.dataparsers.simulator_dataparser import SimulatorDataParserConfig
from nerfstudio.data.dataparsers.pandaset_dataparser import PandaSetDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE

DEFAULT_DISTORTION_PARAMS = np.array([-0.10086563, 0.06112929, -0.04727966, 0.00974163], dtype=np.float32)
BOX_EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
]


def _compute_box_corners_local(dims: np.ndarray) -> np.ndarray:
    """Return the eight local-space corners for a box with [length, width, height]."""
    half_l, half_w, half_h = dims[0] / 2, dims[1] / 2, dims[2] / 2
    return np.array(
        [
            [-half_w, -half_l, -half_h],
            [+half_w, -half_l, -half_h],
            [+half_w, +half_l, -half_h],
            [-half_w, +half_l, -half_h],
            [-half_w, -half_l, +half_h],
            [+half_w, -half_l, +half_h],
            [+half_w, +half_l, +half_h],
            [-half_w, +half_l, +half_h],
        ],
        dtype=np.float32,
    )


@dataclass
class VisualizerConfig:
    """Configuration for dataparser visualizer."""

    data: Path = Path("/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/Simulator_1Nov2025")
    """Directory specifying location of data."""
    
    agent_id: str = "agent_1"
    """ID of the agent to load."""
    
    start_frame: int = 1
    """Start frame"""
    
    end_frame: Optional[int] = None
    """End frame (None = all frames)"""
    
    cameras: Tuple[Literal["camera_1", "camera_2", "camera_3", "camera_4", "all"], ...] = (
        "camera_1",
        "camera_2", 
        "camera_3",
        "camera_4",
    )
    """Which cameras to visualize."""
    
    lidars: Tuple[Literal["lidar_1"], ...] = ("lidar_1",)
    """Which lidars to visualize."""
    
    camera_frustum_scale: float = 0.5
    """Scale of camera frustums in the visualization."""
    
    image_thumbnail_size: int = 100
    """Size of image thumbnails displayed in camera frustums (in pixels)."""
    
    point_size: float = 0.02
    """Size of lidar points in the visualization."""
    
    max_points_per_cloud: Optional[int] = 50000
    """Maximum number of points to display per lidar scan (None = all points)."""

    bounding_box_edge_thickness: float = 0.03
    """Thickness of solid edges used for actor bounding boxes."""

    max_projection_points: Optional[int] = 150000
    """Maximum number of lidar points kept for camera projection diagnostics (None = all)."""

    projection_point_radius_px: int = 2
    """Radius (in pixels) for projected LiDAR points when overlayed on images."""
    
    projection_preview_max_dim: int = 800
    """Maximum dimension (width or height) for projection preview images in pixels."""
    
    timestamp_match_tolerance: float = 1e-3
    """Maximum allowed difference (seconds) when matching timestamps for actor snapshots."""
    
    show_trajectories: bool = False
    """Whether to show actor trajectories as bounding boxes (all frames)."""
    
    port: int = 7007
    """Port for the viser server."""
    
    host: str = "0.0.0.0"
    """Host for the viser server."""


class DataParserVisualizer:
    """Visualizer for dataparser outputs using viser."""

    def __init__(self, config: VisualizerConfig):
        """Initialize the visualizer.
        
        Args:
            config: Configuration for the visualizer
        """
        self.config = config
        
        # Create viser server
        CONSOLE.print(f"[bold green]Starting viser server on {config.host}:{config.port}")
        self.server = viser.ViserServer(host=config.host, port=config.port)
        
        # Load dataparser
        CONSOLE.print(f"[bold green]Loading dataparser from {config.data}")
        dataparser_config = SimulatorDataParserConfig(
            data=config.data,
            agent_id=config.agent_id,
            start_frame=config.start_frame,
            end_frame=config.end_frame,
            cameras=config.cameras,
            lidars=config.lidars,
        )
        self.dataparser = dataparser_config.setup()
        
        
        # Parse data
        CONSOLE.print("[bold green]Parsing data...")
        self.outputs = self.dataparser.get_dataparser_outputs(split="train")
        
        CONSOLE.print(f"[green]✓ Loaded {len(self.outputs.cameras)} cameras")
        if self.outputs.metadata and "lidars" in self.outputs.metadata:
            CONSOLE.print(f"[green]✓ Loaded {len(self.outputs.metadata['lidars'])} lidar scans")
        
        # Storage for visualization handles
        self.camera_handles = {}
        self.lidar_handles = {}
        self.trajectory_handles = {}
        self.image_cache: Dict[int, torch.Tensor] = {}
        self._projection_point_buffer: List[torch.Tensor] = []
        self._projection_points: Optional[torch.Tensor] = None
        self._projection_colors: Optional[np.ndarray] = None
        self._total_projection_points: int = 0
        # Store lidar point clouds and timestamps for per-camera projection
        self._lidar_point_clouds: List[torch.Tensor] = []  # Point clouds in lidar frame
        self._lidar_timestamps: Optional[torch.Tensor] = None  # Timestamps for each lidar scan
        self._lidar_poses: Optional[torch.Tensor] = None  # Lidar poses (3x4) for interpolation
        self._projection_summary_handle = None
        self._projection_image_handle = None
        self._actor_projection_summary_handle = None
        self._actor_projection_image_handle = None
        self._camera_intrinsics_overrides: Dict[int, Dict[str, float]] = {}
        self._timestamp_slider = None
        self._timestamp_summary_handle = None
        self._show_snapshot_checkbox = None
        self._timestamp_snapshot_handles: Dict[int, List[viser.SceneNodeHandle]] = {}  # traj_idx -> list of edge handles
        self._timestamp_values = self._collect_available_timestamps()
        self._actor_projection_intro_text = (
            "[small][bold]Actor bounding box validation[/bold]\n"
            "Select a camera to overlay tracked actors that fall inside the field of view at that timestamp."
            "[/small]"
        )
        self._actor_label_font = ImageFont.load_default()
        
        # Add UI controls
        self._setup_ui()
        self._setup_projection_panel()
        self._setup_timestamp_slider()
        
    def _setup_ui(self):
        """Setup UI controls."""
        with self.server.add_gui_folder("Visibility"):
            self.show_cameras_button = self.server.add_gui_checkbox(
                "Show Cameras",
                initial_value=True,
            )
            self.show_lidars_button = self.server.add_gui_checkbox(
                "Show LiDAR",
                initial_value=True,
            )
            self.show_trajectories_button = self.server.add_gui_checkbox(
                "Show Trajectories",
                initial_value=self.config.show_trajectories,
            )
        
        # Add update callbacks
        @self.show_cameras_button.on_update
        def _(_):
            self._toggle_cameras()
        
        @self.show_lidars_button.on_update
        def _(_):
            self._toggle_lidars()
        
        @self.show_trajectories_button.on_update
        def _(_):
            self._toggle_trajectories()

    def _setup_projection_panel(self):
        """Setup UI for camera-lidar projection diagnostics."""
        with self.server.add_gui_folder("Camera Inspection"):
            tabs = self.server.add_gui_tab_group()
            with tabs.add_tab("LiDAR Overlay"):
                self._projection_intro_text = (
                    "[small][bold]Camera / LiDAR validation[/bold]\n"
                    "Click any camera frustum to reproject the cached LiDAR points into the selected image."
                    "[/small]"
                )
                self._projection_summary_handle = self.server.add_gui_markdown(self._projection_intro_text)
                self._projection_image_handle = self.server.add_gui_markdown("")
            with tabs.add_tab("Actor Boxes"):
                self._actor_projection_summary_handle = self.server.add_gui_markdown(
                    self._actor_projection_intro_text
                )
                self._actor_projection_image_handle = self.server.add_gui_markdown("")

    def _set_projection_summary(self, content: str) -> None:
        """Update the summary markdown for projection panel."""
        if self._projection_summary_handle is not None:
            self._projection_summary_handle.content = content

    def _set_projection_image(self, markdown: str) -> None:
        """Update the markdown panel that displays projection image."""
        if self._projection_image_handle is not None:
            self._projection_image_handle.content = markdown

    def _set_actor_projection_summary(self, content: str) -> None:
        """Update the actor bounding box summary markdown."""
        if self._actor_projection_summary_handle is not None:
            self._actor_projection_summary_handle.content = content

    def _set_actor_projection_image(self, markdown: str) -> None:
        """Update the actor bounding box preview markdown."""
        if self._actor_projection_image_handle is not None:
            self._actor_projection_image_handle.content = markdown

    def _reset_projection_panel(self) -> None:
        """Clear the projection widgets."""
        self._set_projection_summary(self._projection_intro_text)
        self._set_projection_image("")
        self._set_actor_projection_summary(self._actor_projection_intro_text)
        self._set_actor_projection_image("")

    def _setup_timestamp_slider(self) -> None:
        """Create UI for selecting timestamp snapshots."""
        if not self._timestamp_values:
            return
        max_index = len(self._timestamp_values) - 1
        with self.server.add_gui_folder("Actor Snapshot"):
            self._show_snapshot_checkbox = self.server.add_gui_checkbox(
                "Show Actor Boxes",
                initial_value=True,
                hint="Toggle visibility of actor bounding boxes at selected timestamp."
            )
            self._timestamp_slider = self.server.add_gui_slider(
                "Timestamp Index",
                min=0,
                max=max_index,
                step=1,
                initial_value=0,
                hint="Select which timestamp to visualize actor bounding boxes for.",
            )
            self._timestamp_summary_handle = self.server.add_gui_markdown("")

        @self._show_snapshot_checkbox.on_update  # type: ignore[union-attr]
        def _(_: viser.GuiEvent) -> None:
            visible = self._show_snapshot_checkbox.value  # type: ignore[union-attr]
            for edge_handles in self._timestamp_snapshot_handles.values():
                for handle in edge_handles:
                    handle.visible = visible

        @self._timestamp_slider.on_update  # type: ignore[union-attr]
        def _(_: viser.GuiEvent) -> None:
            idx = int(self._timestamp_slider.value)  # type: ignore[union-attr]
            idx = max(0, min(idx, max_index))
            timestamp = self._timestamp_values[idx]
            self._update_timestamp_snapshot(timestamp, idx)

        # Initialize with first timestamp
        self._update_timestamp_snapshot(self._timestamp_values[0], 0)

    def _collect_available_timestamps(self) -> List[float]:
        """Gather sorted unique timestamps from trajectory metadata."""
        metadata = self.outputs.metadata or {}
        trajectories = metadata.get("actor_trajectories") or metadata.get("trajectories")
        if not trajectories:
            return []
        timestamp_values: List[float] = []
        for trajectory in trajectories:
            timestamps = trajectory.get("timestamps")
            if timestamps is None:
                continue
            if isinstance(timestamps, torch.Tensor):
                ts = timestamps.detach().cpu().numpy()
            else:
                ts = np.asarray(timestamps, dtype=np.float64)
            timestamp_values.append(ts)
        if not timestamp_values:
            return []
        concatenated = np.concatenate(timestamp_values, axis=0)
        unique_sorted = np.unique(concatenated)
        return unique_sorted.tolist()

    def _set_timestamp_summary(self, content: str) -> None:
        """Update timestamp snapshot summary."""
        if self._timestamp_summary_handle is not None:
            self._timestamp_summary_handle.content = content
    
    def _create_wireframe_box(
        self, name: str, dims: np.ndarray, wxyz: np.ndarray, position: np.ndarray, color: tuple
    ) -> List[viser.SceneNodeHandle]:
        """Create a wireframe box using thin solid edges.
        
        Args:
            name: Base name for the box segments
            dims: Box dimensions [length, width, height]
            wxyz: Rotation quaternion
            position: Position of box center
            color: RGB color tuple (0-255)
            
        Returns:
            List of line segment handles
        """
        # Actor frame convention: x-right, y-forward, z-up
        corners_local = _compute_box_corners_local(dims)
        
        # Transform corners to world coordinates
        from scipy.spatial.transform import Rotation as R
        quat_scipy = R.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])  # xyzw format
        corners_world = quat_scipy.apply(corners_local) + position
        
        # Define 12 edges of the box (pairs of corner indices)
        edges = [
            # Bottom face
            (0, 1), (1, 2), (2, 3), (3, 0),
            # Top face
            (4, 5), (5, 6), (6, 7), (7, 4),
            # Vertical edges
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        edge_radius = max(1e-3, float(self.config.bounding_box_edge_thickness))
        handles = []
        for edge_idx, (start_idx, end_idx) in enumerate(edges):
            start_point = corners_world[start_idx]
            end_point = corners_world[end_idx]

            edge_vec = end_point - start_point
            edge_length = float(np.linalg.norm(edge_vec))
            if edge_length < 1e-6:
                continue

            direction = edge_vec / edge_length
            up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            if abs(np.dot(direction, up_hint)) > 0.99:
                up_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            y_axis = np.cross(up_hint, direction)
            y_norm = np.linalg.norm(y_axis)
            if y_norm < 1e-6:
                y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                y_axis /= y_norm
            z_axis = np.cross(direction, y_axis)

            rotation_matrix = np.stack([direction, y_axis, z_axis], axis=1)
            edge_rotation = vtf.SO3.from_matrix(rotation_matrix)
            edge_position = (start_point + end_point) / 2.0

            handle = self.server.add_box(
                name=f"{name}/edge_{edge_idx:02d}",
                dimensions=np.array([edge_length, edge_radius, edge_radius]),
                wxyz=edge_rotation.wxyz,
                position=edge_position,
                color=color,
            )
            handles.append(handle)
        
        return handles

    def _compute_box_corners_world(self, dims: np.ndarray, pose: np.ndarray) -> np.ndarray:
        """Compute world-space corners given dims [length, width, height] and SE3 pose."""
        corners_local = _compute_box_corners_local(dims)
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        corners_world = corners_local @ rotation.T + translation
        return corners_world

    def _update_timestamp_snapshot(self, timestamp: float, index: int) -> None:
        """Update actor bounding boxes for a specific timestamp."""
        metadata = self.outputs.metadata or {}
        trajectories = metadata.get("actor_trajectories") or metadata.get("trajectories")
        if not trajectories:
            self._set_timestamp_summary("[yellow]⚠ No trajectory data available for snapshots.")
            return

        tolerance = self.config.timestamp_match_tolerance
        visible = self._show_snapshot_checkbox.value if self._show_snapshot_checkbox else True
        matches = 0

        # First, hide all existing boxes
        for edge_handles in self._timestamp_snapshot_handles.values():
            for handle in edge_handles:
                handle.visible = False

        for traj_idx, trajectory in enumerate(trajectories):
            timestamps = trajectory.get("timestamps")
            poses = trajectory.get("poses")
            dims = trajectory.get("dims")
            if timestamps is None or poses is None or dims is None:
                continue

            if isinstance(timestamps, torch.Tensor):
                ts_tensor = timestamps.to(torch.float64)
            else:
                ts_tensor = torch.tensor(timestamps, dtype=torch.float64)

            if ts_tensor.numel() == 0:
                continue

            diffs = torch.abs(ts_tensor - timestamp)
            min_diff, min_idx = torch.min(diffs, dim=0)
            if min_diff.item() > tolerance:
                # Actor doesn't exist at this timestamp, already hidden above
                continue

            pose_tensor = poses[min_idx]
            if isinstance(pose_tensor, torch.Tensor):
                pose_np = pose_tensor.detach().cpu().numpy()
            else:
                pose_np = np.asarray(pose_tensor, dtype=np.float32)

            if isinstance(dims, torch.Tensor):
                dims_np = dims.detach().cpu().numpy()
            else:
                dims_np = np.asarray(dims, dtype=np.float32)

            # Fix dimension ordering: viser expects [length, width, height] 
            # but our data is [width, length, height], so swap first two
            dims_viser = np.array([dims_np[1], dims_np[0], dims_np[2]], dtype=np.float32)

            rotation = vtf.SO3.from_matrix(pose_np[:3, :3])
            color = self._get_trajectory_color(trajectory.get("label", "unknown"), bool(trajectory.get("is_ego", False)))

            # Update or create wireframe box
            if traj_idx in self._timestamp_snapshot_handles:
                # Remove old wireframe box
                for handle in self._timestamp_snapshot_handles[traj_idx]:
                    handle.remove()
            
            # Create new wireframe box at the updated position
            edge_handles = self._create_wireframe_box(
                name=f"/timestamp_snapshot/actor_{traj_idx:04d}",
                dims=dims_viser,
                wxyz=rotation.wxyz,
                position=pose_np[:3, 3],
                color=color,
            )
            
            # Set visibility
            for handle in edge_handles:
                handle.visible = visible
            
            self._timestamp_snapshot_handles[traj_idx] = edge_handles
            matches += 1

        timestamp_str = f"{timestamp:.3f}"
        summary = f"[green]Snapshot[/green] idx {index} · t={timestamp_str}s · actors shown: {matches}"
        if matches == 0:
            summary = f"[yellow]⚠ No actors found near t={timestamp_str}s (tolerance {tolerance:.3e}s)"
        self._set_timestamp_summary(summary)

    def _accumulate_projection_points(self, point_cloud_world: torch.Tensor) -> None:
        """Accumulate lidar points (in world frame) for later projections."""
        if point_cloud_world is None or point_cloud_world.numel() == 0:
            return
        self._projection_point_buffer.append(point_cloud_world.detach().cpu())

    def _finalize_projection_points(self) -> None:
        """Finalize the cache of LiDAR points used for projections."""
        if not self._projection_point_buffer:
            return
        concatenated = torch.cat(self._projection_point_buffer, dim=0)
        self._total_projection_points = concatenated.shape[0]
        if self.config.max_projection_points is not None and concatenated.shape[0] > self.config.max_projection_points:
            idx = torch.randperm(concatenated.shape[0])[: self.config.max_projection_points]
            concatenated = concatenated[idx]
        intensities = concatenated[:, 3].cpu().numpy()
        colors = (intensity_to_rgb(intensities) * 255).astype(np.uint8)
        self._projection_points = concatenated
        self._projection_colors = colors
        self._projection_point_buffer = []

    def _get_projection_points(self) -> Optional[torch.Tensor]:
        """Return cached LiDAR points (world frame) used for projections."""
        if self._projection_points is None and self._projection_point_buffer:
            self._finalize_projection_points()
        return self._projection_points
    
    def _get_lidar_points_for_camera(self, camera_idx: int) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor], float, float]]:
        """Get lidar points from nearest scan to camera timestamp, with interpolated pose.
        
        Args:
            camera_idx: Index of the camera
            
        Returns:
            Tuple of (points_world, intensities, nearest_time, camera_time) or None if no lidar data available
        """
        if not self._lidar_point_clouds or self._lidar_timestamps is None or self._lidar_poses is None:
            return None
        
        # Get camera timestamp
        cameras = self.outputs.cameras
        if camera_idx >= len(cameras):
            return None
        
        camera_time = cameras.times[camera_idx].item()
        
        # Find nearest lidar scan
        time_diffs = torch.abs(self._lidar_timestamps - camera_time)
        nearest_idx = torch.argmin(time_diffs).item()
        nearest_time = self._lidar_timestamps[nearest_idx].item()
        
        # Interpolate lidar pose at camera timestamp
        from nerfstudio.utils.poses import vectorized_interpolate
        query_time = torch.tensor([[camera_time]], dtype=torch.float64)
        interpolated_l2w = vectorized_interpolate(
            self._lidar_poses,
            self._lidar_timestamps.unsqueeze(-1),
            query_time
        ).squeeze(0)  # [3, 4]
        
        # Get the nearest lidar point cloud (in lidar frame)
        point_cloud_lidar = self._lidar_point_clouds[nearest_idx]
        
        # Downsample if needed
        if self.config.max_projection_points is not None and point_cloud_lidar.shape[0] > self.config.max_projection_points:
            idx = torch.randperm(point_cloud_lidar.shape[0])[: self.config.max_projection_points]
            point_cloud_lidar = point_cloud_lidar[idx]
        
        # Transform points from lidar frame to world frame using interpolated pose
        # transform_points only transforms xyz (first 3 columns), preserving other columns like intensity
        from nerfstudio.cameras.lidars import transform_points
        point_cloud_world = transform_points(point_cloud_lidar, interpolated_l2w.unsqueeze(0)).squeeze(0)
        
        # Extract intensities (if available) - these will be indexed by sequential indices after filtering
        intensities = None
        if point_cloud_world.shape[1] > 3:
            intensities = point_cloud_world[:, 3]
        
        return point_cloud_world, intensities, nearest_time, camera_time
    
    def _toggle_cameras(self):
        """Toggle camera visibility."""
        visible = self.show_cameras_button.value
        for handle in self.camera_handles.values():
            handle.visible = visible
    
    def _toggle_lidars(self):
        """Toggle lidar visibility."""
        visible = self.show_lidars_button.value
        for handle in self.lidar_handles.values():
            handle.visible = visible
    
    def _toggle_trajectories(self):
        """Toggle trajectory visibility."""
        visible = self.show_trajectories_button.value
        for handles in self.trajectory_handles.values():
            for handle in handles:
                handle.visible = visible
    
    def _load_image(self, image_path):
        """Load an image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image as torch tensor (H x W x C) in range [0, 1], or None if failed
        """
        from pathlib import Path
        import torchvision.transforms.functional as TF
        from PIL import Image
        
        try:
            image_path = Path(image_path)
            if not image_path.exists():
                return None
            
            # Load image using PIL
            pil_image = Image.open(image_path)
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to torch tensor (C x H x W) in range [0, 1]
            image_tensor = TF.to_tensor(pil_image)
            
            # Permute to (H x W x C)
            image_tensor = image_tensor.permute(1, 2, 0)
            
            return image_tensor
        except Exception:
            return None

    def _get_image_tensor(self, camera_idx: int) -> Optional[torch.Tensor]:
        """Return cached (optionally undistorted) torch tensor for requested camera index."""
        if camera_idx in self.image_cache:
            return self.image_cache[camera_idx]
        image_filenames = self.outputs.image_filenames
        if not image_filenames or camera_idx >= len(image_filenames):
            return None
        image_tensor = self._load_image(image_filenames[camera_idx])
        if image_tensor is None:
            return None

        image_np = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        undistorted_np, overrides = self._undistort_image(image_np, camera_idx)
        undistorted_tensor = torch.from_numpy(undistorted_np.astype(np.float32) / 255.0)

        self.image_cache[camera_idx] = undistorted_tensor
        self._camera_intrinsics_overrides[camera_idx] = overrides

        return undistorted_tensor

    def _undistort_image(self, image_rgb: np.ndarray, camera_idx: int) -> Tuple[np.ndarray, Dict[str, float]]:
        """Undistort an RGB image using the camera's intrinsic/distortion params."""
        cameras = self.outputs.cameras
        fx = float(cameras.fx[camera_idx].item())
        fy = float(cameras.fy[camera_idx].item())
        cx = float(cameras.cx[camera_idx].item())
        cy = float(cameras.cy[camera_idx].item())
        height, width = image_rgb.shape[:2]

        overrides = {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
        }

        distortion_params = DEFAULT_DISTORTION_PARAMS.copy()
        if cameras.distortion_params is not None:
            dist_tensor = cameras.distortion_params[camera_idx]
            if dist_tensor is not None and torch.any(dist_tensor != 0):
                dist_np = dist_tensor.squeeze().cpu().numpy().astype(np.float32)
                if dist_np.size >= 4:
                    distortion_params = dist_np[:4]

        if distortion_params is None or distortion_params.size < 4:
            return image_rgb, overrides

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, distortion_params, (width, height), 0)
        undistorted_bgr = cv2.undistort(image_bgr, K, distortion_params, None, new_K)
        undistorted_rgb = cv2.cvtColor(undistorted_bgr, cv2.COLOR_BGR2RGB)

        overrides.update(
            {
                "fx": float(new_K[0, 0]),
                "fy": float(new_K[1, 1]),
                "cx": float(new_K[0, 2]),
                "cy": float(new_K[1, 2]),
                "width": undistorted_rgb.shape[1],
                "height": undistorted_rgb.shape[0],
            }
        )

        return undistorted_rgb, overrides
    
    def _resize_image(self, image_tensor, max_size=100):
        """Resize image for display in frustum.
        
        Args:
            image_tensor: Image as torch tensor (H x W x C) in range [0, 1]
            max_size: Maximum dimension size
            
        Returns:
            Resized image as numpy array (H x W x C) uint8
        """
        import torchvision.transforms.functional as TF
        
        # Convert to uint8 and permute to (C x H x W) for torchvision
        image_uint8 = (image_tensor * 255).type(torch.uint8)
        image_uint8 = image_uint8.permute(2, 0, 1)
        
        # Resize
        image_uint8 = TF.resize(image_uint8, max_size, antialias=None)
        
        # Permute back to (H x W x C) and convert to numpy
        image_uint8 = image_uint8.permute(1, 2, 0)
        image_uint8 = image_uint8.cpu().numpy()
        
        return image_uint8

    def _image_tensor_to_uint8(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Convert normalized tensor image to uint8 numpy RGB."""
        return (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

    def _encode_image_markdown(self, image_np: np.ndarray, title: str) -> str:
        """Encode image as markdown data URI, resized to fit projection_preview_max_dim."""
        max_dim = self.config.projection_preview_max_dim
        pil_image = Image.fromarray(image_np)
        # Preserve aspect ratio while capping the max dimension
        pil_image.thumbnail((max_dim, max_dim), Image.BILINEAR)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        safe_title = title.replace("\n", " ")
        return f"![{safe_title}](data:image/png;base64,{encoded})"

    def _render_projection_preview(
        self, image_tensor: torch.Tensor, px: torch.Tensor, py: torch.Tensor, colors: np.ndarray
    ) -> str:
        """Overlay projected points and return markdown-ready data URI."""
        image_np = (image_tensor.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
        draw = ImageDraw.Draw(pil_image)
        radius = max(1, self.config.projection_point_radius_px)
        for x, y, color in zip(px.tolist(), py.tolist(), colors.tolist()):
            x0 = max(0, x - radius)
            y0 = max(0, y - radius)
            x1 = min(image_np.shape[1] - 1, x + radius)
            y1 = min(image_np.shape[0] - 1, y + radius)
            draw.ellipse((x0, y0, x1, y1), fill=tuple(int(c) for c in color), outline=None)
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"![Projected LiDAR](data:image/png;base64,{encoded})"

    def _handle_camera_click(self, camera_idx: int) -> None:
        """Handle camera click events by showing LiDAR projection."""
        try:
            self._project_points_to_image(camera_idx)
        except Exception as exc:
            self._set_projection_summary(f"[red]Projection failed:[/red] {exc}")
            self._set_projection_image("")
        try:
            self._project_actor_boxes_to_image(camera_idx)
        except Exception as exc:
            self._set_actor_projection_summary(f"[red]Actor overlay failed:[/red] {exc}")
            self._set_actor_projection_image("")

    def _project_points_to_image(self, camera_idx: int) -> None:
        """Project nearest lidar scan points into the requested camera image."""
        # Get lidar points from nearest scan with interpolated pose
        result = self._get_lidar_points_for_camera(camera_idx)
        if result is None:
            self._set_projection_summary("[yellow]⚠ No LiDAR data available for projection.")
            self._set_projection_image("")
            return
        
        projection_points, intensities, nearest_time, camera_time = result
        if projection_points is None or projection_points.shape[0] == 0:
            self._set_projection_summary("[yellow]⚠ No LiDAR points available for projection.")
            self._set_projection_image("")
            return

        image_tensor = self._get_image_tensor(camera_idx)
        if image_tensor is None:
            self._set_projection_summary(f"[yellow]⚠ Missing image for camera index {camera_idx}.")
            self._set_projection_image("")
            return

        cameras = self.outputs.cameras
        num_cameras = cameras.camera_to_worlds.shape[0]
        if camera_idx >= num_cameras:
            self._set_projection_summary(f"[yellow]⚠ Invalid camera index {camera_idx}.")
            self._set_projection_image("")
            return

        c2w = cameras.camera_to_worlds[camera_idx].cpu()
        # Use the same inverse function as the reference implementation
        from nerfstudio.utils.poses import inverse
        w2c = inverse(c2w.unsqueeze(0)).squeeze(0)

        points_world = projection_points[:, :3].float()
        # Transform points from world to camera coordinates using transform_points for consistency
        from nerfstudio.cameras.lidars import transform_points
        cam_points = transform_points(points_world, w2c.unsqueeze(0)).squeeze(0)

        # Check if points are in front of camera (in NeRF Studio: forward = -Z, so valid points have Z < 0)
        valid_points = cam_points[:, 2] < 0
        if not torch.any(valid_points):
            self._set_projection_summary(f"[yellow]⚠ Camera {camera_idx} sees no LiDAR points in front.")
            self._set_projection_image("")
            return

        cam_points = cam_points[valid_points]
        color_indices = torch.arange(points_world.shape[0], dtype=torch.int64)[valid_points]

        # Convert from NeRF Studio (forward = -Z, up = +Y) to OpenCV coordinates (forward = +Z, up = -Y)
        cam_points = cam_points.clone()
        cam_points[:, 1] *= -1.0
        cam_points[:, 2] *= -1.0

        depths = cam_points[:, 2]
        depth_mask = depths > 1e-6
        if not torch.any(depth_mask):
            self._set_projection_summary(f"[yellow]⚠ Camera {camera_idx} sees no LiDAR points.")
            self._set_projection_image("")
            return

        cam_points = cam_points[depth_mask]
        depths = depths[depth_mask]
        # Filter color_indices to match the depth_mask (preserve the valid_points filtering)
        color_indices = color_indices[depth_mask]

        overrides = self._camera_intrinsics_overrides.get(camera_idx)
        if overrides:
            fx = overrides["fx"]
            fy = overrides["fy"]
            cx = overrides["cx"]
            cy = overrides["cy"]
            width = int(overrides["width"])
            height = int(overrides["height"])
        else:
            fx = float(cameras.fx[camera_idx].item())
            fy = float(cameras.fy[camera_idx].item())
            cx = float(cameras.cx[camera_idx].item())
            cy = float(cameras.cy[camera_idx].item())
            width = int(cameras.width[camera_idx].item())
            height = int(cameras.height[camera_idx].item())

        u = fx * (cam_points[:, 0] / depths) + cx
        v = fy * (cam_points[:, 1] / depths) + cy

        bounds_mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        if not torch.any(bounds_mask):
            self._set_projection_summary(f"[yellow]⚠ No LiDAR points fall inside camera {camera_idx}'s image.")
            self._set_projection_image("")
            return

        u = u[bounds_mask].round().to(torch.int64)
        v = v[bounds_mask].round().to(torch.int64)
        color_indices = color_indices[bounds_mask]

        # Use intensities from the lidar scan
        # color_indices are sequential indices into the filtered point cloud (after valid_points and depth_mask)
        # So we can use them directly to index into intensities
        if intensities is not None:
            # color_indices are indices into the filtered intensities array
            intensities_filtered = intensities[color_indices].cpu().numpy()
            colors = (intensity_to_rgb(intensities_filtered) * 255).astype(np.uint8)
        else:
            # Fallback: use default color if no intensities
            colors = np.full((len(color_indices), 3), 255, dtype=np.uint8)

        markdown_image = self._render_projection_preview(image_tensor, u, v, colors)
        self._set_projection_image(markdown_image)
        num_points = len(color_indices)
        total_points = projection_points.shape[0]
        time_diff = abs(camera_time - nearest_time)
        self._set_projection_summary(
            f"[green]Camera {camera_idx:05d}[/green] · showing {num_points} / {total_points} points "
            f"from nearest lidar scan (Δt={time_diff:.3f}s)"
        )

    def _project_actor_boxes_to_image(self, camera_idx: int) -> None:
        """Overlay actor bounding boxes visible to the selected camera."""
        metadata = self.outputs.metadata or {}
        trajectories = metadata.get("actor_trajectories") or metadata.get("trajectories")
        if not trajectories:
            self._set_actor_projection_summary("[yellow]⚠ No trajectory data available.")
            self._set_actor_projection_image("")
            return

        cameras = self.outputs.cameras
        if camera_idx >= len(cameras):
            self._set_actor_projection_summary(f"[yellow]⚠ Invalid camera index {camera_idx}.")
            self._set_actor_projection_image("")
            return

        if cameras.times is None:
            self._set_actor_projection_summary("[yellow]⚠ Camera timestamps missing; cannot align actors.")
            self._set_actor_projection_image("")
            return

        camera_time = cameras.times[camera_idx].item()
        image_tensor = self._get_image_tensor(camera_idx)
        if image_tensor is None:
            self._set_actor_projection_summary(f"[yellow]⚠ Missing image for camera index {camera_idx}.")
            self._set_actor_projection_image("")
            return

        overrides = self._camera_intrinsics_overrides.get(camera_idx)
        if overrides:
            fx = overrides["fx"]
            fy = overrides["fy"]
            cx = overrides["cx"]
            cy = overrides["cy"]
            width = int(overrides["width"])
            height = int(overrides["height"])
        else:
            fx = float(cameras.fx[camera_idx].item())
            fy = float(cameras.fy[camera_idx].item())
            cx = float(cameras.cx[camera_idx].item())
            cy = float(cameras.cy[camera_idx].item())
            width = int(cameras.width[camera_idx].item())
            height = int(cameras.height[camera_idx].item())

        c2w = cameras.camera_to_worlds[camera_idx].cpu()
        from nerfstudio.utils.poses import inverse

        w2c = inverse(c2w.unsqueeze(0)).squeeze(0)
        tolerance = self.config.timestamp_match_tolerance

        image_np = self._image_tensor_to_uint8(image_tensor)
        pil_image = Image.fromarray(image_np.copy())
        draw = ImageDraw.Draw(pil_image)

        matches = 0
        considered = 0
        max_time_delta = 0.0

        for traj_idx, trajectory in enumerate(trajectories):
            timestamps = trajectory.get("timestamps")
            poses = trajectory.get("poses")
            dims = trajectory.get("dims")
            if timestamps is None or poses is None or dims is None:
                continue

            if isinstance(timestamps, torch.Tensor):
                ts_tensor = timestamps.to(torch.float64)
            else:
                ts_tensor = torch.tensor(timestamps, dtype=torch.float64)
            if ts_tensor.numel() == 0:
                continue

            diffs = torch.abs(ts_tensor - camera_time)
            min_diff, min_idx = torch.min(diffs, dim=0)
            time_error = float(min_diff.item())
            if time_error > tolerance:
                continue

            considered += 1
            max_time_delta = max(max_time_delta, time_error)

            pose_index = int(min_idx.item()) if isinstance(min_idx, torch.Tensor) else int(min_idx)
            pose_entry = poses[pose_index]
            if isinstance(pose_entry, torch.Tensor):
                pose_np = pose_entry.detach().cpu().numpy()
            else:
                pose_np = np.asarray(pose_entry, dtype=np.float32)

            if isinstance(dims, torch.Tensor):
                dims_np = dims.detach().cpu().numpy()
            else:
                dims_np = np.asarray(dims, dtype=np.float32)
            dims_viser = np.array([dims_np[1], dims_np[0], dims_np[2]], dtype=np.float32)

            # Compute world corners and project into camera
            corners_world = self._compute_box_corners_world(dims_viser, pose_np)
            corners_world_tensor = torch.from_numpy(corners_world).to(device=w2c.device, dtype=w2c.dtype)
            cam_points = transform_points(corners_world_tensor, w2c.unsqueeze(0)).squeeze(0)
            forward_mask = cam_points[:, 2] < 0
            if not torch.any(forward_mask):
                continue

            cam_points = cam_points.clone()
            cam_points[:, 1] *= -1.0
            cam_points[:, 2] *= -1.0
            depths = cam_points[:, 2]

            pixel_coords: List[Optional[Tuple[int, int]]] = []
            inside_fov = False
            for corner_idx in range(cam_points.shape[0]):
                depth = float(depths[corner_idx].item())
                if depth <= 1e-6:
                    pixel_coords.append(None)
                    continue
                u = fx * (cam_points[corner_idx, 0].item() / depth) + cx
                v = fy * (cam_points[corner_idx, 1].item() / depth) + cy
                if u < 0 or u >= width or v < 0 or v >= height:
                    pixel_coords.append(None)
                    continue
                pixel_coords.append((int(round(u)), int(round(v))))
                inside_fov = True

            if not inside_fov:
                continue

            color = self._get_trajectory_color(trajectory.get("label", "unknown"), bool(trajectory.get("is_ego", False)))
            for start_idx, end_idx in BOX_EDGES:
                p0 = pixel_coords[start_idx]
                p1 = pixel_coords[end_idx]
                if p0 is None or p1 is None:
                    continue
                draw.line([p0[0], p0[1], p1[0], p1[1]], fill=color, width=2)

            valid_pixels = [pt for pt in pixel_coords if pt is not None]
            if valid_pixels:
                min_u = min(pt[0] for pt in valid_pixels)
                min_v = min(pt[1] for pt in valid_pixels)
                label = trajectory.get("label") or f"actor_{traj_idx:04d}"
                text = label if not trajectory.get("is_ego", False) else f"{label} (ego)"
                text_bbox = draw.textbbox((0, 0), text, font=self._actor_label_font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
                text_x = int(np.clip(min_u, 0, max(0, width - text_w - 1)))
                text_y = int(np.clip(min_v - text_h - 2, 0, max(0, height - text_h - 1)))
                draw.rectangle(
                    (text_x - 2, text_y - 2, text_x + text_w + 2, text_y + text_h + 2),
                    fill=(0, 0, 0),
                )
                draw.text((text_x, text_y), text, font=self._actor_label_font, fill=(255, 255, 255))

            matches += 1

        if matches == 0:
            self._set_actor_projection_summary(
                f"[yellow]⚠ No actors intersect camera {camera_idx:05d}'s FOV at t={camera_time:.3f}s (tol={tolerance:.3e}s)."
            )
            self._set_actor_projection_image("")
            return

        overlay_np = np.array(pil_image)
        markdown_image = self._encode_image_markdown(overlay_np, f"Actor boxes · camera {camera_idx:05d}")
        self._set_actor_projection_image(markdown_image)
        self._set_actor_projection_summary(
            "[green]Actor overlay[/green] · "
            f"camera {camera_idx:05d} shows {matches} actor(s) (checked {considered} within tolerance, "
            f"max Δt={max_time_delta:.3e}s, tol={tolerance:.3e}s)"
        )
    
    def visualize_cameras(self):
        """Visualize camera frustums with images."""
        CONSOLE.print("[bold blue]Visualizing cameras...")
        cameras = self.outputs.cameras
        image_filenames = self.outputs.image_filenames
        
        for idx in range(len(cameras)):
            # Get camera pose
            c2w = cameras.camera_to_worlds[idx].cpu().numpy()
            
            # Convert rotation to viser format
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            
            # Calculate FOV from intrinsics and image dimensions
            fx = cameras.fx[idx].item()
            fy = cameras.fy[idx].item()
            width = int(cameras.width[idx].item())
            height = int(cameras.height[idx].item())
            # Horizontal FOV: 2 * arctan(width / (2 * fx))
            fov = float(2 * np.arctan(width / (2 * fx)))
            aspect = float(width / height)
            
            # Load and process image
            image_uint8 = None
            if image_filenames and idx < len(image_filenames):
                try:
                    image = self._get_image_tensor(idx)
                    if image is not None:
                        # Resize for display (similar to viewer.py)
                        image_uint8 = self._resize_image(image, max_size=self.config.image_thumbnail_size)
                except Exception as e:
                    CONSOLE.print(f"[yellow]⚠ Could not load image {idx}: {e}")
            
            # Add camera frustum
            camera_handle = self.server.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=fov,
                scale=self.config.camera_frustum_scale,
                aspect=aspect,
                image=image_uint8,
                wxyz=R.wxyz,
                position=c2w[:3, 3],
            )
            
            # Add click handler to move view to camera
            @camera_handle.on_click
            def _(
                event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle], cam_idx: int = idx
            ) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz
                self._handle_camera_click(cam_idx)
            
            self.camera_handles[idx] = camera_handle
        
        CONSOLE.print(f"[green]✓ Visualized {len(self.camera_handles)} camera frustums")
    
    def visualize_lidars(self):
        """Visualize lidar point clouds and poses."""
        if not self.outputs.metadata or "lidars" not in self.outputs.metadata:
            CONSOLE.print("[yellow]⚠ No lidar data found")
            return
        
        CONSOLE.print("[bold blue]Visualizing lidar point clouds...")
        
        lidars = self.outputs.metadata["lidars"]
        point_clouds = self.outputs.metadata.get("point_clouds", [])
        
        if not point_clouds:
            CONSOLE.print("[yellow]⚠ No point clouds found in metadata")
            return
        
        # Store lidar poses and timestamps for interpolation
        lidar_poses_list = []
        lidar_times_list = []
        
        for idx in range(len(lidars)):
            # Get lidar pose
            l2w_tensor = lidars.lidar_to_worlds[idx].cpu()
            l2w = l2w_tensor.numpy()
            
            # Store pose and timestamp for interpolation
            lidar_poses_list.append(l2w_tensor)
            if lidars.times is not None:
                lidar_times_list.append(lidars.times[idx].item())
            else:
                # Fallback: use index as timestamp if times not available
                lidar_times_list.append(float(idx))
            
            # Add coordinate frame for lidar pose
            R = vtf.SO3.from_matrix(l2w[:3, :3])
            self.server.add_frame(
                name=f"/lidar_frames/lidar_{idx:05d}",
                wxyz=R.wxyz,
                position=l2w[:3, 3],
                axes_length=0.5,
                axes_radius=0.02,
            )
            
            # Store point cloud in lidar frame (for per-camera projection)
            point_cloud = point_clouds[idx]
            self._lidar_point_clouds.append(point_cloud.cpu())
            
            # Transform and visualize point cloud (for 3D visualization)
            point_cloud_world = transform_points(point_cloud, l2w_tensor.unsqueeze(0)).squeeze(0)
            self._accumulate_projection_points(point_cloud_world)
            
            # Downsample if needed
            max_points = self.config.max_points_per_cloud
            if max_points is not None and point_cloud_world.shape[0] > max_points:
                indices = torch.randperm(point_cloud_world.shape[0])[:max_points]
                point_cloud_world = point_cloud_world[indices]
            
            # Extract xyz and intensity
            points_xyz = point_cloud_world[:, :3].cpu().numpy()
            intensity = point_cloud_world[:, 3].cpu().numpy()
            colors = intensity_to_rgb(intensity)
            
            # Add point cloud
            lidar_handle = self.server.add_point_cloud(
                name=f"/lidar/lidar_{idx:05d}",
                points=points_xyz,
                colors=colors,
                point_size=self.config.point_size,
                point_shape="circle",
            )
            
            self.lidar_handles[idx] = lidar_handle
        
        # Store lidar poses and timestamps for interpolation
        if lidar_poses_list:
            self._lidar_poses = torch.stack(lidar_poses_list)  # [N, 3, 4]
            self._lidar_timestamps = torch.tensor(lidar_times_list, dtype=torch.float64)  # [N]
        
        CONSOLE.print(f"[green]✓ Visualized {len(self.lidar_handles)} lidar scans")
        self._finalize_projection_points()
    
    def visualize_trajectories(self):
        """Visualize actor trajectories as bounding boxes."""
        if not self.config.show_trajectories:
            return
        
        metadata = self.outputs.metadata or {}
        trajectories = metadata.get("actor_trajectories") or metadata.get("trajectories")
        if not trajectories:
            CONSOLE.print("[yellow]⚠ No trajectory data found")
            return
        
        CONSOLE.print("[bold blue]Visualizing actor trajectories...")
        
        for traj_idx, trajectory in enumerate(trajectories):
            poses = trajectory["poses"]  # T x 4 x 4
            dims = trajectory["dims"]  # [width, length, height]
            label = trajectory.get("label", "unknown")
            is_ego = bool(trajectory.get("is_ego", False))
            
            if isinstance(dims, torch.Tensor):
                dims_np = dims.detach().cpu().numpy()
            else:
                dims_np = np.asarray(dims, dtype=np.float32)
            dims_viser = np.array([dims_np[1], dims_np[0], dims_np[2]], dtype=np.float32)

            # Create handles for each pose in the trajectory
            traj_handles = []
            
            for pose_idx, pose in enumerate(poses):
                pose_np = pose.cpu().numpy()
                R = vtf.SO3.from_matrix(pose_np[:3, :3])
                position = pose_np[:3, 3]
                
                # Add bounding box
                box_handle = self.server.add_box(
                    name=f"/trajectories/traj_{traj_idx:04d}/pose_{pose_idx:04d}",
                    dimensions=dims_viser,
                    wxyz=R.wxyz,
                    position=position,
                    color=self._get_trajectory_color(label, is_ego=is_ego),
                )
                traj_handles.append(box_handle)
            
            self.trajectory_handles[traj_idx] = traj_handles
        
        CONSOLE.print(f"[green]✓ Visualized {len(self.trajectory_handles)} actor trajectories")
    
    def _get_trajectory_color(self, label: str, is_ego: bool = False) -> tuple[int, int, int]:
        """Get color for trajectory based on label.
        
        Args:
            label: Object label/type
            is_ego: Whether this is the ego vehicle
            
        Returns:
            RGB color tuple (0-255)
        """
        if is_ego:
            return (255, 255, 255)

        # Define colors for different object types
        color_map = {
            "TYPE_VEHICLE": (0, 255, 0),
            "TYPE_SMALL_CAR": (0, 200, 50),
            "TYPE_MEDIUM_CAR": (0, 180, 80),
            "TYPE_COMPACT_CAR": (0, 160, 100),
            "TYPE_LUXURY_CAR": (0, 140, 120),
            "TYPE_HEAVY_TRUCK": (255, 0, 0),
            "TYPE_BUS": (255, 100, 0),
            "TYPE_SEMITRACTOR": (255, 150, 0),
            "TYPE_SEMITRAILER": (255, 200, 0),
            "TYPE_MOTORBIKE": (0, 255, 255),
            "TYPE_PEDESTRIAN": (255, 0, 255),
            "TYPE_DUMMY_BICYCLE": (0, 0, 255),
            "TYPE_DUMMY_CYCLIST": (100, 0, 255),
            "TYPE_MOTORCYCLE_RIDER": (150, 0, 255),
        }
        return color_map.get(label, (128, 128, 128))  # Default gray
    
    def visualize_scene_box(self):
        """Visualize the scene bounding box."""
        if self.outputs.scene_box is None:
            return
        
        scene_box = self.outputs.scene_box
        aabb = scene_box.aabb
        
        # Calculate center and dimensions
        center = (aabb[0] + aabb[1]) / 2
        dimensions = aabb[1] - aabb[0]
        
        # Add bounding box (semi-transparent gray)
        self.server.add_box(
            name="/scene/bounding_box",
            dimensions=dimensions.cpu().numpy(),
            position=center.cpu().numpy(),
            color=(100, 100, 100),
        )
        
        CONSOLE.print("[green]✓ Visualized scene bounding box")
    
    def run(self):
        """Run the visualizer."""
        # Visualize all components
        self.visualize_cameras()
        self.visualize_lidars()
        self.visualize_trajectories()
        self.visualize_scene_box()
        
        CONSOLE.print("\n[bold green]✓ Visualization complete!")
        CONSOLE.print(f"[bold cyan]Open your browser to: http://localhost:{self.config.port}")
        CONSOLE.print("[yellow]Press Ctrl+C to exit\n")
        
        # Keep the server running
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            CONSOLE.print("\n[bold red]Shutting down...")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    config = tyro.cli(VisualizerConfig)
    
    visualizer = DataParserVisualizer(config)
    visualizer.run()


if __name__ == "__main__":
    entrypoint()

