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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

import torch
from rich.console import Console

from nerfstudio.cameras.lidars import transform_points
from nerfstudio.data.datamanagers.image_lidar_datamanager import (
    ImageLidarDataManager,
    ImageLidarDataManagerConfig,
    ImageLidarDataProcessor,
    _cache_images,
    _cache_points,
    lidar_packed_collate,
)
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.pixel_samplers import ScaledPatchSamplerConfig
from nerfstudio.data.utils.data_utils import remove_dynamic_points
from nerfstudio.data.utils.dataloaders import (
    CacheDataloader,
    FixedIndicesEvalDataloader,
    RandIndicesEvalDataloader,
)
from nerfstudio.model_components.ray_generators import RayGenerator


CONSOLE = Console(width=120)


@dataclass
class ADDataManagerConfig(ImageLidarDataManagerConfig):
    """A basic data manager"""

    _target: Type = field(default_factory=lambda: ADDataManager)
    """Target class to instantiate."""
    train_num_lidar_rays_per_batch: int = 16384
    """Number of lidar rays per batch to use per training iteration."""
    train_num_rays_per_batch: int = 40960
    """Number of camera rays per batch to use per training iteration (equals 40 32x32 patches)."""
    eval_num_lidar_rays_per_batch: int = 8192
    """Number of lidar rays per batch to use per eval iteration."""
    eval_num_rays_per_batch: int = 40960
    """Number of camera rays per batch to use per eval iteration (equals 40 32x32 patches)."""
    downsample_factor: float = 1
    """Downsample factor for the lidar. If <1, downsample will be used."""
    image_divisible_by: int = 1
    """If >1, images will be cropped to be divisible by this number."""
    pixel_sampler: ScaledPatchSamplerConfig = field(default_factory=ScaledPatchSamplerConfig)
    """AD models default to path-based pixel sampler."""


class ADDataManager(ImageLidarDataManager):
    """This extends the VanillaDataManager to support lidar data.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: ADDataManagerConfig

    def create_eval_dataset(self) -> InputDataset:
        dataset = super().create_eval_dataset()
        # Maybe crop images
        cams = dataset.cameras
        for width in cams.width.unique().tolist():
            width_crop = _find_smallest_crop(width, self.config.image_divisible_by)
            cams.width[cams.width == width] = width - width_crop
        for height in cams.height.unique().tolist():
            height_crop = _find_smallest_crop(height, self.config.image_divisible_by)
            cams.height[cams.height == height] = height - height_crop
        return dataset

    def change_patch_sampler(self, patch_scale: int, patch_size: int):
        """Change the camera sample to sample rays in NxN patches."""
        if self.config.train_num_rays_per_batch % (patch_size**2) != 0:
            CONSOLE.print("[bold yellow]WARNING: num_rays should be divisible by patch_size^2.")
        if patch_scale == self.eval_pixel_sampler.patch_scale and patch_size == self.eval_pixel_sampler.patch_size:
            return

        # Change train
        if self.use_mp:
            for func_queue in self.func_queues:
                func_queue.put((_worker_change_patch_sampler, (patch_scale, patch_size), {}))
            self.clear_data_queue()  # remove any old, invalid, batch
        else:
            _worker_change_patch_sampler(self.data_procs[0], patch_scale, patch_size)

        # Change eval
        self.eval_pixel_sampler.patch_scale = patch_scale
        self.eval_pixel_sampler.patch_size = patch_size
        if patch_scale % 2 == 0 and self.eval_ray_generator.image_coords[0, 0, 0] == 0.5:
            self.eval_ray_generator.image_coords = self.eval_ray_generator.image_coords - 0.5

    def get_accumulated_lidar_points(self, remove_dynamic: bool = False):
        """Get the lidar points for the current batch."""
        lidars = self.train_lidar_dataset.lidars
        point_clouds = self.train_lidar_dataset.point_clouds
        if remove_dynamic:
            assert "trajectories" in self.train_lidar_dataset.metadata, "No trajectories found in dataset."

            point_clouds = remove_dynamic_points(
                point_clouds,
                lidars.lidar_to_worlds,
                lidars.times,
                self.train_lidar_dataset.metadata["trajectories"],
            )

        return torch.cat(
            [transform_points(pc[:, :3], l2w) for pc, l2w in zip(point_clouds, lidars.lidar_to_worlds)], dim=0
        )

    

    def update_dataset_resolution(self, new_scale_factor: float):
        """
        Updates the dataset resolution, rescales cameras, clears caches,
        and restarts the dataloader to propagate changes to workers.
        
        This function handles the complete resolution update process:
        1. Updates camera intrinsics (fx, fy, cx, cy, width, height)
        2. Updates dataset scale_factor (used when loading images)
        3. Recreates cached images at new resolution
        4. Restarts parallel data processors with new cached images
        5. Updates pixel samplers and ray generators
        """
        
        # 2. Get current scale factor
        old_scale_factor = self.train_dataset.scale_factor
        
        # Avoid doing work if scale hasn't changed
        if abs(new_scale_factor - old_scale_factor) < 1e-6:
            return

        print(f"[DataManager] Updating resolution. Scale: {old_scale_factor} -> {new_scale_factor}")

        # 3. Calculate relative scale for camera intrinsics
        # rescale_output_resolution multiplies by the scaling factor, so we need
        # to calculate the ratio from old to new
        rescale_ratio = new_scale_factor / old_scale_factor
        
        # 4. Update Camera Intrinsics (fx, fy, cx, cy, width, height)
        self.train_dataset.cameras.rescale_output_resolution(rescale_ratio)
        
        # 5. Update the dataset's internal scale factor
        # This is used by get_numpy_image() to resize images when loading
        self.train_dataset.scale_factor = new_scale_factor
        
        # 5b. Also update eval dataset if it exists
        if hasattr(self, "eval_dataset") and self.eval_dataset is not None:
            eval_old_scale = self.eval_dataset.scale_factor
            eval_rescale_ratio = new_scale_factor / eval_old_scale
            self.eval_dataset.cameras.rescale_output_resolution(eval_rescale_ratio)
            self.eval_dataset.scale_factor = new_scale_factor

        # 6. Handle ImageLidarDataManager (parallel data processors)
        if hasattr(self, "data_procs") and self.data_procs is not None:
            # Stop existing data processors
            if self.use_mp:
                for proc in self.data_procs:
                    proc.terminate()
                    proc.join()
            
            # Clear data queue to remove any stale batches
            self.clear_data_queue()
            
            # Recreate pixel sampler with updated dataset
            self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
            
            # Recreate cached images at new resolution
            # This is critical - images must be reloaded at the new scale_factor
            cached_images = _cache_images(self.train_dataset, self.config.max_thread_workers, self.config.collate_fn)
            cached_points = _cache_points(self.train_lidar_dataset, self.config.max_thread_workers, lidar_packed_collate)
            
            # Recreate data processors with new cached images
            self.data_procs = [
                ImageLidarDataProcessor(
                    out_queue=self.data_queue,
                    func_queue=func_queue,
                    config=self.config,
                    dataparser_outputs=self.train_dataparser_outputs,
                    image_dataset=self.train_dataset,
                    pixel_sampler=self.train_pixel_sampler,
                    lidar_dataset=self.train_lidar_dataset,
                    point_sampler=self.train_point_sampler,
                    cached_images=cached_images,
                    cached_points=cached_points,
                )
                for func_queue in self.func_queues
            ]
            
            # Restart processes
            if self.use_mp:
                for proc in self.data_procs:
                    proc.start()
                print("[DataManager] Restarted data processes with new resolution")
        
        # 7. Handle eval dataloader (if using CacheDataloader)
        # CacheDataloader caches images in cached_collated_batch during initialization
        # We must recreate it entirely so it reloads images at the new resolution
        if hasattr(self, "eval_image_dataloader") and hasattr(self, "eval_dataset"):
            # Delete old iterator
            if hasattr(self, "iter_eval_image_dataloader"):
                del self.iter_eval_image_dataloader
            
            # Recreate eval dataloader with updated dataset
            # This will reload images at the new resolution
            self.eval_image_dataloader = CacheDataloader(
                self.eval_dataset,
                num_images_to_sample_from=self.config.eval_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.eval_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
                exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
            )
            self.iter_eval_image_dataloader = iter(self.eval_image_dataloader)
            
            # Recreate eval pixel sampler and ray generator
            self.eval_pixel_sampler = self._get_pixel_sampler(self.eval_dataset, self.config.eval_num_rays_per_batch)
            self.eval_ray_generator = RayGenerator(self.eval_dataset.cameras.to(self.device))
            
            # Also recreate eval dataloaders if they exist (from ParallelDataManager.setup_eval)
            if hasattr(self, "fixed_indices_eval_dataloader"):
                self.fixed_indices_eval_dataloader = FixedIndicesEvalDataloader(
                    dataset=self.eval_dataset,
                    device=self.device,
                    num_workers=self.world_size * 4,
                )
            if hasattr(self, "eval_dataloader"):
                self.eval_dataloader = RandIndicesEvalDataloader(
                    dataset=self.eval_dataset,
                    device=self.device,
                    num_workers=self.world_size * 4,
                )
        
        # 8. Handle train dataloader (if using CacheDataloader - for non-parallel case)
        # NOTE: This section is for VanillaDataManager, NOT for ImageLidarDataManager/ADDataManager
        # ADDataManager uses parallel data processors (data_procs) instead of train_image_dataloader
        # So hasattr(self, "train_image_dataloader") will be False for ADDataManager
        # Only recreate if train_image_dataloader exists (non-parallel case)
        if hasattr(self, "train_image_dataloader") and not hasattr(self, "data_procs"):
            # Delete old iterator and dataloader
            if hasattr(self, "iter_train_image_dataloader"):
                del self.iter_train_image_dataloader
            del self.train_image_dataloader
            
            # Recreate train dataloader with updated dataset
            # This will reload images at the new resolution
            self.train_image_dataloader = CacheDataloader(
                self.train_dataset,
                num_images_to_sample_from=self.config.train_num_images_to_sample_from,
                num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
                device=self.device,
                num_workers=self.world_size * 4,
                pin_memory=True,
                collate_fn=self.config.collate_fn,
                exclude_batch_keys_from_device=self.exclude_batch_keys_from_device,
            )
            self.iter_train_image_dataloader = iter(self.train_image_dataloader)
            
            # Recreate train pixel sampler and ray generator
            self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
            self.train_ray_generator = RayGenerator(self.train_dataset.cameras.to(self.device))
        
        # 9. Clear any dataset-level image caches
        if hasattr(self.train_dataset, "image_cache"):
            self.train_dataset.image_cache.clear()
        if hasattr(self, "eval_dataset") and hasattr(self.eval_dataset, "image_cache"):
            self.eval_dataset.image_cache.clear()
        
        print(f"[DataManager] Resolution update complete. New image dimensions: {self.train_dataset.cameras.height[0].item()}x{self.train_dataset.cameras.width[0].item()}")


def _find_smallest_crop(dim: int, divider: int):
    crop_amount = 0
    while dim % divider:
        crop_amount += 1
        dim -= 1
    return crop_amount


def _worker_change_patch_sampler(worker, patch_scale, patch_size):
    worker.pixel_sampler.patch_scale = patch_scale
    worker.pixel_sampler.patch_size = patch_size
    # Ensure ray is generated from patch center (for odd scales center of pixel == center of patch)
    if patch_scale % 2 == 0 and worker.ray_generator.image_coords[0, 0, 0] == 0.5:
        worker.ray_generator.image_coords -= 0.5
