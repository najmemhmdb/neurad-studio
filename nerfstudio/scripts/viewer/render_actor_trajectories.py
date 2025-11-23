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
Utility to render actor trajectories from `SimulatorDataParser` into a video.

Generates a top-down visualization where each actor is represented using a
colored 3D bounding box footprint and its historical trajectory. This is useful
for quickly validating annotations without spinning up the interactive viewer.

Example:
    python nerfstudio/scripts/viewer/render_actor_trajectories.py \
        --data /path/to/simulator \
        --start-frame 1 \
        --end-frame 250 \
        --output-path /tmp/trajectories.mp4
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple

import cv2
import numpy as np
import tyro
import tyro.extras

from nerfstudio.data.dataparsers.simulator_dataparser import SimulatorDataParserConfig
from nerfstudio.utils.rich_utils import CONSOLE


COLOR_MAP: Dict[str, Tuple[int, int, int]] = {
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


@dataclass
class TrajectoryVideoConfig:
    """Configuration for rendering trajectory videos."""

    data: Path = Path("/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/Simulator_1Nov2025")
    """Directory specifying location of simulator data."""

    agent_id: str = "agent_1"
    """Agent ID to visualize."""

    start_frame: int = 1
    """First frame index (inclusive)."""

    end_frame: Optional[int] = None
    """Last frame index (exclusive)."""

    cameras: Tuple[Literal["camera_1", "camera_2", "camera_3", "camera_4", "all"], ...] = (
        "camera_1",
        "camera_2",
        "camera_3",
        "camera_4",
    )
    """Cameras to instruct the dataparser (kept for consistency)."""

    lidars: Tuple[Literal["lidar_1"], ...] = ("lidar_1",)
    """Lidars to instruct the dataparser (kept for consistency)."""

    output_path: Path = Path("actor_trajectories.mp4")
    """Path to the output video file."""

    fps: int = 12
    """Frames per second for the output video."""

    width: int = 1280
    """Output video width in pixels."""

    height: int = 720
    """Output video height in pixels."""

    margin_meters: float = 10.0
    """Extra padding (meters) added around the bounding extents."""

    history_length: int = 30
    """Number of previous timesteps to draw in the trajectory trail (0 = full)."""

    background_color: Tuple[int, int, int] = (15, 15, 15)
    """Background color of the canvas (B, G, R)."""

    thickness_px: int = 2
    """Line thickness for bounding boxes."""


def _world_to_pixel(
    point_xy: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    width: int,
    height: int,
) -> Tuple[int, int]:
    """Project world XY coordinates onto the 2D canvas."""
    ranges = np.maximum(bounds_max - bounds_min, 1e-3)
    normalized = (point_xy - bounds_min) / ranges
    x_px = int(np.clip(normalized[0], 0.0, 1.0) * (width - 1))
    y_px = int((1.0 - np.clip(normalized[1], 0.0, 1.0)) * (height - 1))
    return x_px, y_px


def _compute_box_corners(pose: np.ndarray, dims: np.ndarray) -> np.ndarray:
    """Return 2D XY coordinates of the four box corners projected to the ground plane."""
    width_w, length_w = float(dims[0]), float(dims[1])
    half_l = length_w / 2.0
    half_w = width_w / 2.0
    local_corners = np.array(
        [
            [-half_l, -half_w, 0.0, 1.0],
            [-half_l, half_w, 0.0, 1.0],
            [half_l, half_w, 0.0, 1.0],
            [half_l, -half_w, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    world_corners = (pose @ local_corners.T).T[:, :2]
    return world_corners


def _get_color(label: str, is_ego: bool = False) -> Tuple[int, int, int]:
    """Return BGR color for a label (OpenCV uses BGR)."""
    if is_ego:
        return (255, 255, 255)
    rgb = COLOR_MAP.get(label, (128, 128, 128))
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _collect_frames(trajectories: Iterable[Dict]) -> Tuple[DefaultDict[float, List[Dict]], np.ndarray]:
    """Organize trajectory entries per timestamp and return XY bounds."""
    frames: DefaultDict[float, List[Dict]] = defaultdict(list)
    centers: List[np.ndarray] = []

    for actor_idx, traj in enumerate(trajectories):
        poses = traj["poses"].cpu().numpy()
        timestamps = traj["timestamps"].cpu().numpy()
        dims = traj["dims"].cpu().numpy()
        label = traj.get("label", "unknown")

        for pose, ts in zip(poses, timestamps):
            frames[float(ts)].append(
                {
                    "pose": pose,
                    "dims": dims,
                    "label": label,
                    "actor_id": actor_idx,
                    "is_ego": bool(traj.get("is_ego", False)),
                }
            )
            centers.append(pose[:2, 3])

    if not centers:
        raise RuntimeError("No actor trajectories were found in the metadata.")

    center_array = np.stack(centers, axis=0)
    return frames, center_array


def render_actor_trajectories_video(config: TrajectoryVideoConfig) -> None:
    """Render colored bounding boxes for all actors into a video."""
    dataparser_config = SimulatorDataParserConfig(
        data=config.data,
        agent_id=config.agent_id,
        start_frame=config.start_frame,
        end_frame=config.end_frame,
        cameras=config.cameras,
        lidars=config.lidars,
    )
    dataparser = dataparser_config.setup()
    outputs = dataparser.get_dataparser_outputs(split="train")

    metadata = outputs.metadata or {}
    trajectories = metadata.get("actor_trajectories") or metadata.get("trajectories")
    if trajectories is None:
        raise RuntimeError("Dataparser outputs do not contain any actor trajectory metadata.")
    frames, center_array = _collect_frames(trajectories)

    min_xy = center_array.min(axis=0) - config.margin_meters
    max_xy = center_array.max(axis=0) + config.margin_meters

    timestamps = sorted(frames.keys())
    if not timestamps:
        raise RuntimeError("Unable to find timestamps for trajectories.")

    CONSOLE.print(
        f"[bold green]Rendering {len(timestamps)} timesteps "
        f"({timestamps[0]:.2f}s → {timestamps[-1]:.2f}s) to {config.output_path}"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(config.output_path),
        fourcc,
        max(config.fps, 1),
        (config.width, config.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {config.output_path}")

    history: DefaultDict[int, List[np.ndarray]] = defaultdict(list)

    try:
        for ts in timestamps:
            canvas = np.full((config.height, config.width, 3), config.background_color, dtype=np.uint8)
            actors = frames[ts]
            timestamp_text = f"t = {ts:.2f}s"
            cv2.putText(
                canvas,
                timestamp_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            for actor in actors:
                pose = actor["pose"]
                dims = actor["dims"]
                color = _get_color(actor["label"], is_ego=actor.get("is_ego", False))

                corners = _compute_box_corners(pose, dims)
                pixel_corners = np.array(
                    [_world_to_pixel(pt, min_xy, max_xy, config.width, config.height) for pt in corners],
                    dtype=np.int32,
                )

                cv2.polylines(canvas, [pixel_corners], isClosed=True, color=color, thickness=config.thickness_px)
                center_xy = pose[:2, 3]
                history[actor["actor_id"]].append(center_xy)

                if config.history_length > 0 and len(history[actor["actor_id"]]) > config.history_length:
                    history[actor["actor_id"]] = history[actor["actor_id"]][-config.history_length :]

                history_points = np.array(
                    [_world_to_pixel(pt, min_xy, max_xy, config.width, config.height) for pt in history[actor["actor_id"]]],
                    dtype=np.int32,
                )
                if history_points.shape[0] >= 2:
                    cv2.polylines(canvas, [history_points], isClosed=False, color=color, thickness=1)

            writer.write(canvas)
    finally:
        writer.release()

    CONSOLE.print(f"[bold cyan]✓ Saved trajectory video to {config.output_path}")


def entrypoint() -> None:
    """CLI entrypoint."""
    tyro.extras.set_accent_color("bright_yellow")
    config = tyro.cli(TrajectoryVideoConfig)
    render_actor_trajectories_video(config)


if __name__ == "__main__":
    entrypoint()

