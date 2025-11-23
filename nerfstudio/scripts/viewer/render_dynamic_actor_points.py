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
Render a top-down video showing only LiDAR points belonging to dynamic actors.

This script filters each LiDAR sweep using the per-frame actor cuboids emitted
by `SimulatorDataParser`. Only points that fall inside dynamic (non-stationary)
actors' oriented bounding boxes are kept, and are color-coded by actor label.

Example:
    python nerfstudio/scripts/viewer/render_dynamic_actor_points.py \
        --data /path/to/simulator \
        --output-path /tmp/dynamic_points.mp4
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import tyro
import tyro.extras

from nerfstudio.cameras.lidars import transform_points
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
class DynamicPointsConfig:
    """Configuration for rendering dynamic actor point clouds."""

    data: Path = Path("/mnt/public/Ehsan/datasets/private/Najmeh/simulated_data/Simulator_1Nov2025")
    agent_id: str = "agent_1"
    start_frame: int = 1
    end_frame: Optional[int] = None
    cameras: Tuple[Literal["camera_1", "camera_2", "camera_3", "camera_4", "all"], ...] = (
        "camera_1",
        "camera_2",
        "camera_3",
        "camera_4",
    )
    lidars: Tuple[Literal["lidar_1"], ...] = ("lidar_1",)

    output_path: Path = Path("dynamic_actor_points.mp4")
    fps: int = 12
    width: int = 1280
    height: int = 720
    margin_meters: float = 10.0
    point_radius_px: int = 1
    max_points_per_actor: Optional[int] = 5000
    include_ego: bool = False
    timestamp_precision: int = 4
    per_actor_output_dir: Optional[Path] = None

    background_color: Tuple[int, int, int] = (10, 10, 10)


def _timestamp_key(value: float, precision: int) -> float:
    """Round timestamps to a consistent precision for dict lookups."""
    return round(value, precision)


def _world_to_pixel(
    xy: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    width: int,
    height: int,
) -> Tuple[int, int]:
    ranges = np.maximum(bounds_max - bounds_min, 1e-3)
    normalized = (xy - bounds_min) / ranges
    x_px = int(np.clip(normalized[0], 0.0, 1.0) * (width - 1))
    y_px = int((1.0 - np.clip(normalized[1], 0.0, 1.0)) * (height - 1))
    return x_px, y_px


def _label_to_bgr(label: str) -> Tuple[int, int, int]:
    rgb = COLOR_MAP.get(label, (200, 200, 200))
    return int(rgb[2]), int(rgb[1]), int(rgb[0])


def _collect_dynamic_actor_frames(
    trajectories: List[Dict],
    precision: int,
    include_ego: bool,
) -> Tuple[DefaultDict[float, List[Dict]], Optional[np.ndarray], Dict[str, Dict[str, str]]]:
    frames: DefaultDict[float, List[Dict]] = defaultdict(list)
    centers: List[np.ndarray] = []
    actor_registry: Dict[str, Dict[str, str]] = {}

    for traj_idx, traj in enumerate(trajectories):
        if bool(traj.get("stationary", False)):
            continue
        if not include_ego and bool(traj.get("is_ego", False)):
            continue

        actor_id = (
            str(
                traj.get("object_id")
                or traj.get("uuid")
                or traj.get("id")
                or f"actor_{traj_idx:04d}"
            )
        )
        poses = traj["poses"].float()
        dims = traj["dims"].float()
        timestamps = traj["timestamps"].float()
        label = traj.get("label", "unknown")

        actor_registry[actor_id] = {"label": label}

        for pose, ts in zip(poses, timestamps):
            ts_key = _timestamp_key(float(ts.item()), precision)
            frames[ts_key].append(
                {
                    "pose": pose.clone(),
                    "dims": dims.clone(),
                    "label": label,
                    "actor_id": actor_id,
                }
            )
            centers.append(pose[:2, 3].cpu().numpy())

    if centers:
        return frames, np.stack(centers, axis=0), actor_registry
    return frames, None, actor_registry


def _extract_actor_points(
    point_cloud_world: torch.Tensor,
    actors: List[Dict],
    max_points_per_actor: Optional[int],
) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    if not actors or point_cloud_world.numel() == 0:
        return []

    points_xyz = point_cloud_world[:, :3]
    ones = torch.ones((points_xyz.shape[0], 1), dtype=points_xyz.dtype, device=points_xyz.device)
    points_h = torch.cat([points_xyz, ones], dim=1)
    filtered: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

    for actor in actors:
        pose = actor["pose"]
        dims = actor["dims"]
        world_to_actor = torch.linalg.inv(pose)
        local_points = (points_h @ world_to_actor.T)[:, :3]
        half_dims = dims / 2.0
        inside = torch.all(torch.abs(local_points) <= half_dims, dim=1)
        if not inside.any():
            continue
        actor_points = points_xyz[inside].detach().cpu().numpy()
        if max_points_per_actor is not None and actor_points.shape[0] > max_points_per_actor:
            choice = np.random.choice(actor_points.shape[0], max_points_per_actor, replace=False)
            actor_points = actor_points[choice]
        filtered.append((actor_points, _label_to_bgr(actor["label"])))

    return filtered


def _render_video(
    samples: List[Dict],
    frames: DefaultDict[float, List[Dict]],
    config: DynamicPointsConfig,
    min_xy: np.ndarray,
    max_xy: np.ndarray,
    output_path: Path,
    actor_id: Optional[str] = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(config.fps, 1),
        (config.width, config.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer at {output_path}")

    try:
        for sample in samples:
            actors = frames.get(sample["timestamp"], [])
            if actor_id is not None:
                actors = [a for a in actors if a.get("actor_id") == actor_id]

            canvas = np.full((config.height, config.width, 3), config.background_color, dtype=np.uint8)

            if actors:
                cloud_world = transform_points(sample["cloud"], sample["transform"])
                actor_points = _extract_actor_points(
                    cloud_world,
                    actors,
                    config.max_points_per_actor,
                )

                for points_xyz, color in actor_points:
                    if points_xyz.size == 0:
                        continue
                    pixels = np.array(
                        [_world_to_pixel(pt[:2], min_xy, max_xy, config.width, config.height) for pt in points_xyz],
                        dtype=np.int32,
                    )
                    for x_px, y_px in pixels:
                        cv2.circle(canvas, (x_px, y_px), config.point_radius_px, color, thickness=-1)

            actor_suffix = f" · actor: {actor_id}" if actor_id is not None else ""
            timestamp_text = f"t = {sample['timestamp']:.3f}s · actors: {len(actors)}{actor_suffix}"
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

            writer.write(canvas)
    finally:
        writer.release()


def render_dynamic_actor_points_video(config: DynamicPointsConfig) -> None:
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
    lidars = metadata.get("lidars")
    point_clouds = metadata.get("point_clouds")

    if trajectories is None:
        raise RuntimeError("Dataparser outputs do not contain actor trajectory metadata.")
    if lidars is None or point_clouds is None:
        raise RuntimeError("Dataparser outputs do not contain lidar point clouds.")

    dynamic_frames, centers, actor_registry = _collect_dynamic_actor_frames(
        trajectories,
        precision=config.timestamp_precision,
        include_ego=config.include_ego,
    )

    if centers is None or centers.size == 0:
        raise RuntimeError("No dynamic actors found to visualize.")

    min_xy = centers.min(axis=0) - config.margin_meters
    max_xy = centers.max(axis=0) + config.margin_meters

    samples = []
    for idx, (timestamp, transform, cloud) in enumerate(
        zip(lidars.times, lidars.lidar_to_worlds, point_clouds)
    ):
        ts_key = _timestamp_key(float(timestamp.item()), config.timestamp_precision)
        tf = torch.eye(4, dtype=cloud.dtype)
        tf[:3, :4] = transform
        samples.append(
            {
                "timestamp": ts_key,
                "transform": tf,
                "cloud": cloud.clone(),
            }
        )

    _render_video(
        samples=samples,
        frames=dynamic_frames,
        config=config,
        min_xy=min_xy,
        max_xy=max_xy,
        output_path=config.output_path,
        actor_id=None,
    )
    CONSOLE.print(f"[bold cyan]✓ Saved dynamic actor point video to {config.output_path}")

    if config.per_actor_output_dir:
        for actor_id, meta in actor_registry.items():
            label = meta.get("label", "actor")
            safe_label = label.lower().replace(" ", "_")
            per_actor_path = config.per_actor_output_dir / f"{actor_id}_{safe_label}.mp4"
            _render_video(
                samples=samples,
                frames=dynamic_frames,
                config=config,
                min_xy=min_xy,
                max_xy=max_xy,
                output_path=per_actor_path,
                actor_id=actor_id,
            )
        CONSOLE.print(
            f"[bold green]✓ Wrote {len(actor_registry)} per-actor videos to {config.per_actor_output_dir}"
        )


def entrypoint() -> None:
    tyro.extras.set_accent_color("bright_yellow")
    config = tyro.cli(DynamicPointsConfig)
    render_dynamic_actor_points_video(config)


if __name__ == "__main__":
    entrypoint()

