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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro
import viser
import viser.transforms as vtf

from nerfstudio.cameras.lidars import intensity_to_rgb, transform_points
from nerfstudio.data.dataparsers.simulator_dataparser import SimulatorDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE


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
    
    show_trajectories: bool = True
    """Whether to show actor trajectories as bounding boxes."""
    
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
        
        # Add UI controls
        self._setup_ui()
        
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
            
            # Calculate FOV from intrinsics
            fx = cameras.fx[idx].item()
            cx = cameras.cx[idx].item()
            cy = cameras.cy[idx].item()
            fov = float(2 * np.arctan(cx / fx))
            aspect = float(cx / cy)
            
            # Load and process image
            image_uint8 = None
            if image_filenames and idx < len(image_filenames):
                try:
                    image = self._load_image(image_filenames[idx])
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
            def _(event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz
            
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
        
        for idx in range(len(lidars)):
            # Get lidar pose
            l2w = lidars.lidar_to_worlds[idx].cpu().numpy()
            
            # Add coordinate frame for lidar pose
            R = vtf.SO3.from_matrix(l2w[:3, :3])
            self.server.add_frame(
                name=f"/lidar_frames/lidar_{idx:05d}",
                wxyz=R.wxyz,
                position=l2w[:3, 3],
                axes_length=0.5,
                axes_radius=0.02,
            )
            
            # Transform and visualize point cloud
            point_cloud = point_clouds[idx]
            point_cloud_world = transform_points(point_cloud, l2w)
            
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
        
        CONSOLE.print(f"[green]✓ Visualized {len(self.lidar_handles)} lidar scans")
    
    def visualize_trajectories(self):
        """Visualize actor trajectories as bounding boxes."""
        if not self.config.show_trajectories:
            return
        
        if not self.outputs.metadata or "actor_trajectories" not in self.outputs.metadata:
            CONSOLE.print("[yellow]⚠ No trajectory data found")
            return
        
        CONSOLE.print("[bold blue]Visualizing actor trajectories...")
        
        trajectories = self.outputs.metadata["actor_trajectories"]
        
        for traj_idx, trajectory in enumerate(trajectories):
            poses = trajectory["poses"]  # T x 4 x 4
            dims = trajectory["dims"]  # [width, length, height]
            label = trajectory.get("label", "unknown")
            
            # Create handles for each pose in the trajectory
            traj_handles = []
            
            for pose_idx, pose in enumerate(poses):
                pose_np = pose.cpu().numpy()
                R = vtf.SO3.from_matrix(pose_np[:3, :3])
                position = pose_np[:3, 3]
                
                # Add bounding box
                box_handle = self.server.add_box(
                    name=f"/trajectories/traj_{traj_idx:04d}/pose_{pose_idx:04d}",
                    dimensions=dims.cpu().numpy(),
                    wxyz=R.wxyz,
                    position=position,
                    color=self._get_trajectory_color(label),
                )
                traj_handles.append(box_handle)
            
            self.trajectory_handles[traj_idx] = traj_handles
        
        CONSOLE.print(f"[green]✓ Visualized {len(self.trajectory_handles)} actor trajectories")
    
    def _get_trajectory_color(self, label: str) -> tuple[int, int, int]:
        """Get color for trajectory based on label.
        
        Args:
            label: Object label/type
            
        Returns:
            RGB color tuple (0-255)
        """
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

