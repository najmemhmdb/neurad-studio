# Copyright 2024 the authors of NeuRAD and contributors.
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
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
Pose and Intrinsics Optimizers
"""

from __future__ import annotations
import open3d as o3d
import functools
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type, Union
import json

import numpy
import torch
import tyro
from jaxtyping import Float, Int
from torch import Tensor, nn
from typing_extensions import assert_never
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.lie_groups import *
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.utils import poses as pose_utils
from nerfstudio.data.dataparsers.pandaset_dataparser import _pandaset_pose_to_matrix
from nerfstudio.data.dataparsers.ad_dataparser import OPENCV_TO_NERFSTUDIO
import os
import json
import yaml
import numpy as np



import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless save
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import open3d as o3d
import matplotlib.pyplot as plt 

@dataclass
class CameraOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: Union[Tuple, float] = 1e-2
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for these fields.
    optimizer: tyro.conf.Suppress[Optional[OptimizerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    scheduler: tyro.conf.Suppress[Optional[SchedulerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    def __post_init__(self):
        if self.optimizer is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\noptimizer is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)

        if self.scheduler is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\nscheduler is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


@dataclass
class CameraVelocityOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera velocities."""

    _target: Type = field(default_factory=lambda: CameraVelocityOptimizer)

    enabled: bool = False
    """Optimize velocities"""

    zero_initial_velocities: bool = False
    """Do not use initial velocities in cameras as a starting point"""

    linear_l2_penalty: float = 1e-6
    """L2 penalty on linear velocity"""

    angular_l2_penalty: float = 1e-5
    """L2 penalty on angular velocity"""

class CameraVelocityOptimizer(nn.Module):
    """Layer that modifies camera velocities during training."""

    config: CameraVelocityOptimizerConfig

    def __init__(
        self,
        config: CameraVelocityOptimizerConfig,
        num_cameras: int,
        num_unique_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.num_unique_cameras = num_unique_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices

        # Initialize learnable parameters.
        if self.config.enabled:
            self.linear_velocity_adjustment = torch.nn.Parameter(
                ((torch.rand((num_cameras, 3), device=device) - 0.5) * 1e-1)
            )
            self.angular_velocity_adjustment = torch.nn.Parameter(
                ((torch.rand((num_cameras, 3), device=device) - 0.5) * 1e-4)
            )
            self.time_to_center_pixel_adjustment = torch.nn.Parameter(
                ((torch.rand((num_unique_cameras), device=device) - 0.5) * 1e-6)
            )

    def get_time_to_center_pixel_adjustment(self, camera: Union[Cameras, Lidars]) -> Float[Tensor, "num_cameras 1"]:
        """Get the time to center pixel adjustment."""
        sensor_idx = camera.metadata["sensor_idxs"].view(-1)
        if self.config.enabled:
            return self.time_to_center_pixel_adjustment[sensor_idx]
        return torch.zeros_like(sensor_idx, device=camera.device)

    def apply_to_camera_velocity(self, camera: Union[Cameras, Lidars], return_init_only) -> torch.Tensor:
        init_velocities = None
        sensor_to_world = camera.camera_to_worlds if isinstance(camera, Cameras) else camera.lidar_to_worlds
        if self.config.zero_initial_velocities:
            init_velocities = torch.zeros((len(camera), 6), device=sensor_to_world.device)
        else:
            assert camera.metadata["linear_velocities_local"] is not None
            init_velocities = torch.hstack(
                [camera.metadata["linear_velocities_local"], camera.metadata["angular_velocities_local"]]
            )

        if not self.config.enabled or return_init_only:  # or not self.training:
            return init_velocities

        if camera.metadata is None or "cam_idx" not in camera.metadata:
            return init_velocities

        cam_idx = camera.metadata["cam_idx"]
        adj = torch.cat([self.linear_velocity_adjustment[cam_idx, :], self.angular_velocity_adjustment[cam_idx, :]])[
            None
        ]
        return init_velocities + adj

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.enabled:
            loss_dict["camera_velocity_regularizer"] = (
                self.linear_velocity_adjustment.norm(dim=-1).mean() * self.config.linear_l2_penalty
                + self.angular_velocity_adjustment.norm(dim=-1).mean() * self.config.angular_l2_penalty
            )

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera velocity optimizer metrics"""
        if self.config.enabled:
            lin = self.linear_velocity_adjustment.detach().norm(dim=-1)
            ang = self.angular_velocity_adjustment.detach().norm(dim=-1)
            metrics_dict["camera_opt_vel_max"] = lin.max()
            metrics_dict["camera_opt_vel_mean"] = lin.mean()
            metrics_dict["camera_opt_ang_vel_max"] = ang.max()
            metrics_dict["camera_opt_ang_vel_mean"] = ang.mean()
            for i in range(self.num_unique_cameras):
                metrics_dict[f"camera_opt_ttc_pixel_adjustment_{i}"] = self.time_to_center_pixel_adjustment[i].detach()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        vel_opt_params = list(self.parameters())
        if self.config.enabled:
            assert len(vel_opt_params) > 0
            param_groups["camera_velocity_opt_linear"] = vel_opt_params[0:1]
            param_groups["camera_velocity_opt_angular"] = vel_opt_params[1:2]
            param_groups["camera_velocity_opt_time_to_center_pixel"] = vel_opt_params[2:3]
        else:
            assert len(vel_opt_params) == 0


class CameraOptimizer(nn.Module):
    """Layer that modifies camera poses to be optimized as well as the field during training."""

    config: CameraOptimizerConfig

    def __init__(
        self,
        config: CameraOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices
        os.makedirs(f"calib_results/save/", exist_ok=True)
        # Initialize learnable parameters.
        if self.config.mode == "off":
            pass
        elif self.config.mode in ("SO3xR3", "SE3"):
            self.pose_adjustment = torch.nn.Parameter(torch.zeros((num_cameras, 6), device=device))
            self.step_counter = 0
        else:
            assert_never(self.config.mode)

    def _get_pose_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return self.pose_adjustment

    def forward(
        self,
        indices: Int[Tensor, "camera_indices"],
    ) -> Float[Tensor, "camera_indices 3 4"]:
        """Indexing into camera adjustments.
        Args:
            indices: indices of Cameras to optimize.
        Returns:
            Transformation matrices from optimized camera coordinates
            to given camera coordinates.
        """
        outputs = []

        # Apply learned transformation delta.
        if self.config.mode == "off":
            pass
        elif self.config.mode == "SO3xR3":
            
            self.step_counter += 1
            if self.step_counter % 2000 == 0 or self.step_counter == 1:
                
                # Convert tensor to a JSON-compatible format
                pose_adjustment_list = exp_map_SO3xR3(self.pose_adjustment).detach().cpu().tolist()
                # Save to a JSON file
                path = f"calib_results/save"
                with open(f"{path}/pose_adjustment{self.step_counter}.json", "w") as f:
                    json.dump(pose_adjustment_list, f)
                    
            outputs.append(exp_map_SO3xR3(self._get_pose_adjustment()[indices, :]))
        elif self.config.mode == "SE3":
            outputs.append(exp_map_SE3(self._get_pose_adjustment()[indices, :]))
        else:
            assert_never(self.config.mode)
        # Detach non-trainable indices by setting to identity transform
        if self.non_trainable_camera_indices is not None:
            if self.non_trainable_camera_indices.device != self.pose_adjustment.device:
                self.non_trainable_camera_indices = self.non_trainable_camera_indices.to(self.pose_adjustment.device)
            outputs[0][self.non_trainable_camera_indices] = torch.eye(4, device=self.pose_adjustment.device)[:3, :4]

        # Return: identity if no transforms are needed, otherwise multiply transforms together.
        if len(outputs) == 0:
            # Note that using repeat() instead of tile() here would result in unnecessary copies.
            return torch.eye(4, device=self.device)[None, :3, :4].tile(indices.shape[0], 1, 1)
        return functools.reduce(pose_utils.multiply, outputs)

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            correction_matrices = self(raybundle.camera_indices.squeeze())  # type: ignore
            raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
            raybundle.directions = (
                torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None])
                .squeeze()
                .to(raybundle.origins)
            )


    def apply_to_camera(self, camera: Cameras) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            assert camera.metadata is not None, "Must provide id of camera in its metadata"
            assert "cam_idx" in camera.metadata, "Must provide id of camera in its metadata"
            camera_idx = camera.metadata["cam_idx"]
            adj = self(torch.tensor([camera_idx], dtype=torch.long, device=camera.device))  # type: ignore
            adj = torch.cat([adj, torch.Tensor([0, 0, 0, 1])[None, None].to(adj)], dim=1)
            camera.camera_to_worlds = torch.bmm(camera.camera_to_worlds, adj)

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            pose_adjustment = self._get_pose_adjustment()
            # loss_dict["camera_opt_regularizer"] = (
            #     pose_adjustment[:, :3].norm(dim=-1).mean() * self.config.trans_l2_penalty
            #     + pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
            # )

    def get_correction_matrices(self):
        """Get optimized pose correction matrices"""
        return self(torch.arange(0, self.num_cameras).long())

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            pose_adjustment = self._get_pose_adjustment()
            metrics_dict["camera_opt_translation"] = pose_adjustment[:, :3].norm()
            metrics_dict["camera_opt_rotation"] = pose_adjustment[:, 3:].norm()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        camera_opt_params = list(self.parameters())
        if self.config.mode != "off":
            assert len(camera_opt_params) > 0
            param_groups["camera_opt"] = camera_opt_params
        else:
            assert len(camera_opt_params) == 0


@dataclass
class CameraLidarTemporalOptimizerConfig(InstantiateConfig):
    """Configuration of optimization for camera poses."""

    _target: Type = field(default_factory=lambda: CameraLidarTemporalOptimizer)

    mode: Literal["off", "SO3xR3", "SE3"] = "off"
    """Pose optimization strategy to use. If enabled, we recommend SO3xR3."""

    trans_l2_penalty: Union[Tuple, float] = 1e-2
    """L2 penalty on translation parameters."""

    rot_l2_penalty: float = 1e-3
    """L2 penalty on rotation parameters."""

    # tyro.conf.Suppress prevents us from creating CLI arguments for these fields.
    optimizer: tyro.conf.Suppress[Optional[OptimizerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""

    scheduler: tyro.conf.Suppress[Optional[SchedulerConfig]] = field(default=None)
    """Deprecated, now specified inside the optimizers dict"""
    
    lidar_times: Optional[Tensor] = None
    """Lidar times"""
    lidar2w: Optional[Tensor] = None
    """Lidar to world transform"""

    def __post_init__(self):
        if self.optimizer is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\noptimizer is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)

        if self.scheduler is not None:
            import warnings

            from nerfstudio.utils.rich_utils import CONSOLE

            CONSOLE.print(
                "\nscheduler is no longer specified in the CameraOptimizerConfig, it is now defined with the rest of the param groups inside the config file under the name 'camera_opt'\n",
                style="bold yellow",
            )
            warnings.warn("above message coming from", FutureWarning, stacklevel=3)


    

class CameraLidarTemporalOptimizer(CameraOptimizer):
    """Camera optimizer that optimizes the camera poses and lidar poses temporally."""

    def __init__(self,
        config: CameraLidarTemporalOptimizerConfig,
        num_cameras: int,
        device: Union[torch.device, str],
        non_trainable_camera_indices: Optional[Int[Tensor, "num_non_trainable_cameras"]] = None,
        **kwargs,
    ) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.num_cameras = num_cameras
        self.device = device
        self.non_trainable_camera_indices = non_trainable_camera_indices
        self.step_counter = 0
        self.camera_count = 6
        self.sequence_length = num_cameras // (self.camera_count + 1)
        self.ext_init, ext_noisy = self.load_init_extrinsics() # [num_cams, 6]
        self.extrinsics_trans = torch.nn.Parameter(ext_noisy[:, :3])  # [num_cams, 3]
        self.extrinsics_rot   = torch.nn.Parameter(ext_noisy[:, 3:])  # [num_cams, 3]

        self.offsets = torch.nn.Parameter(torch.tensor([[0.0], [-0.010797], [0.010783], [-0.050045], [-0.031357], [0.03129]]).to(device))
        self.offsets.requires_grad_(False)
        self.lidar2w = kwargs["lidar2w"].cuda()
        self.camera_to_worlds = torch.zeros((self.camera_count*self.sequence_length, 12), device=self.device)
        self.lidar_times = kwargs["lidar_times"].cuda()
        os.makedirs(f"calib_results/save/times", exist_ok=True)
        os.makedirs(f"calib_results/save/adjustments", exist_ok=True)
        self.errors = { 0: {'x': [], 'y': [], 'z': []}, 
                        1: {'x': [], 'y': [], 'z': []}, 
                        2: {'x': [], 'y': [], 'z': []}, 
                        3: {'x': [], 'y': [], 'z': []}, 
                        4: {'x': [], 'y': [], 'z': []}, 
                        5: {'x': [], 'y': [], 'z': []}}
        # self.adjustment = torch.zeros((self.camera_count, 3, 4), device=device)
        eye = torch.eye(3)[None, :3, :4]
        constant = torch.zeros_like(eye[..., :, :1])
        self.mask_adj_rot = torch.cat([eye, constant], dim=-1).to('cuda')
        # self.adjustment = torch.cat([eye, constant], dim=-1).to('cuda')
        # self.active_idx = 0


    # def target_row_this_step(self) -> int:
    #     self.active_idx = (self.step_counter // 50) % self.camera_count
        

    # @torch.no_grad()
    # def mask_extrinsics_grad_to_row(self):
    #     self.target_row_this_step()
    #     if self.extrinsics.grad is None:
    #         return
    #     mask = torch.zeros_like(self.extrinsics.grad)  # shape (6, 6)
    #     mask[self.active_idx].fill_(1.0)
    #     self.extrinsics.grad.mul_(mask)


    def load_init_extrinsics(self):
        l2s_dict = yaml.load(open(os.path.join(os.path.dirname(__file__), "pandaset_extrinsics.yaml"), "r"), Loader=yaml.FullLoader)
        sensors = ["front_camera", "front_left_camera", "front_right_camera", "back_camera","left_camera","right_camera"]
        lidar2sensor_list_gt = []
        lidar2sensor_list_noisy = []
        
        def _skew(w):
            wx, wy, wz = w
            z = torch.zeros((), dtype=w.dtype, device=w.device)
            return torch.tensor([[0., -wz,  wy],
                                [wz,  0., -wx],
                                [-wy, wx,  0.]], dtype=w.dtype, device=w.device)

        def so3_exp(phi):
            # Rodrigues to mirror your exp_map_SO3xR3 rotation block
            theta = torch.linalg.norm(phi)
            I = torch.eye(3, dtype=phi.dtype, device=phi.device)
            if theta < 1e-8:
                K = _skew(phi)
                return I + K + 0.5 * (K @ K)  # 2nd-order for stability
            K = _skew(phi)
            s, c = torch.sin(theta), torch.cos(theta)
            a = s / theta
            b = (1 - c) / (theta * theta)
            return I + a * K + b * (K @ K)

        def rotmat_to_rotvec_match_exp(R):
            """
            Log that matches the Rodrigues used in exp_map_SO3xR3.
            Picks the sign so that so3_exp(phi) ≈ R (not R^T).
            """
            # Standard log (your implementation is fine)
            trace = torch.clamp(torch.trace(R), -1.0, 3.0)
            cos_theta = (trace - 1.0) * 0.5
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            theta = torch.acos(cos_theta)

            if theta < 1e-6:
                rx = 0.5 * (R[2,1] - R[1,2])
                ry = 0.5 * (R[0,2] - R[2,0])
                rz = 0.5 * (R[1,0] - R[0,1])
                phi = torch.stack([rx, ry, rz])
            else:
                s = torch.sin(theta)
                kx = (R[2,1] - R[1,2]) / (2.0 * s)
                ky = (R[0,2] - R[2,0]) / (2.0 * s)
                kz = (R[1,0] - R[0,1]) / (2.0 * s)
                axis = torch.stack([kx, ky, kz])
                axis = axis / (axis.norm() + 1e-12)
                phi = axis * theta

            # Pick the sign that best reconstructs R with our Rodrigues
            err_pos = (so3_exp(phi)   - R).abs().max()
            err_neg = (so3_exp(-phi)  - R).abs().max()
            return phi if err_pos <= err_neg else -phi

        def mat4_to_SO3xR3_twist(T4x4: torch.Tensor) -> torch.Tensor:
            """
            Convert a 4x4 transform to the 6D vector for exp_map_SO3xR3.
            Returns xi = [t_x, t_y, t_z, phi_x, phi_y, phi_z] (shape [6]).
            """
            T4x4 = T4x4.to(dtype=torch.float64)
            R = T4x4[:3, :3]
            t = T4x4[:3, 3]
            # If R may be slightly non-orthonormal, project once:
            # U, _, Vh = torch.linalg.svd(R); R = U @ Vh;  if torch.det(R) < 0: U[:, -1] *= -1; R = U @ Vh
            phi = rotmat_to_rotvec_match_exp(R)
            return torch.cat([t, phi])
                    
        for i,sensor in enumerate(sensors):
            l2s = l2s_dict[sensor]
            l2sensor = {}
            l2sensor["position"] = l2s["extrinsic"]["transform"]["translation"]
            l2sensor["heading"] = l2s["extrinsic"]["transform"]["rotation"]
            l2sensor_4x4 = _pandaset_pose_to_matrix(l2sensor)
            lidar2sensor = torch.from_numpy(l2sensor_4x4[:3, :])
            xi = mat4_to_SO3xR3_twist(lidar2sensor)
            lidar2sensor_list_gt.append(xi)

            l2sensor_4x4_noisy = l2sensor_4x4.copy()
            l2sensor_4x4_noisy[:3, 3]  -= ((0.02 * i) + 0.08)
            lidar2sensor_noisy = torch.from_numpy(l2sensor_4x4_noisy[:3, :])
            xi_noisy = mat4_to_SO3xR3_twist(lidar2sensor_noisy)
            lidar2sensor_list_noisy.append(xi_noisy)
           
        return torch.stack(lidar2sensor_list_gt).to(dtype=torch.float32), torch.stack(lidar2sensor_list_noisy).to(dtype=torch.float32)

    
    def forward(self, all_indices: Int[Tensor, "camera_indices"]) -> Float[Tensor, "camera_indices 3 4"]:
        if self.config.mode == "SO3xR3":
            outputs = []
            mask_ext = all_indices < (self.sequence_length * self.camera_count)
            ext_indices = all_indices[mask_ext]
            lidar_indices = all_indices[~mask_ext]
            self.step_counter += 1

            ext_adjustments = self._get_ext_adjustment()
            time_offsets = self._get_offsets()

            camera_indices = ext_indices // self.sequence_length
            extrinsics = ext_adjustments[camera_indices]
            camera_offsets = time_offsets[camera_indices] 
            seq_indices = ext_indices % self.sequence_length  #[batch size]
            query_times = self.lidar_times[seq_indices] + camera_offsets
            
            extrinsics_mapped = pose_utils.to4x4(exp_map_SO3xR3(extrinsics))
            
            interpolated_batch_lidar2w = pose_utils.vectorized_interpolate(
                                            self.lidar2w, self.lidar_times, query_times
                                        )


            # interpolated_batch_lidar2w = self.lidar2w[seq_indices]
            lidar2w_4x4 = pose_utils.to4x4(interpolated_batch_lidar2w)

            sensor2w = lidar2w_4x4 @ torch.inverse(extrinsics_mapped)
            R = sensor2w[:, :3, :3]
            sensor2w[:, :3, :3] = R @ torch.from_numpy(OPENCV_TO_NERFSTUDIO).cuda().to(dtype=torch.float32)

            # calculate adjustment
            s2w_cameras = self.camera_to_worlds[ext_indices.cpu()].to(sensor2w.device)
            sensor2w_init = s2w_cameras.reshape(s2w_cameras.shape[0], 3, 4)
            init_4x4 = pose_utils.to4x4(sensor2w_init)
            adjustment = sensor2w @ torch.inverse(init_4x4) 
            outputs.append(adjustment[:, :3, :4])
            # outputs.append(torch.eye(4, device=adjustment.device)[None, :3, :4].tile(ext_indices.shape[0], 1, 1))
            if lidar_indices.any():
                outputs.append(torch.eye(4, device=adjustment.device)[None, :3, :4].tile(lidar_indices.shape[0], 1, 1))
            self.adjustment = adjustment[:, :3, :4]

            # visualization and debug information
            with torch.no_grad():
                camera_indices_unique = torch.unique(camera_indices.clone().detach())
                for idx in camera_indices_unique:
                    idx_mask = (ext_indices.clone().detach() // self.sequence_length) == idx
                    self.errors[idx.item()]['x'].append([self.step_counter, self.extrinsics_trans[idx.item()][0].item() - self.ext_init[idx.item()][0].item()])
                    self.errors[idx.item()]['y'].append([self.step_counter, self.extrinsics_trans[idx.item()][1].item() - self.ext_init[idx.item()][1].item()])
                    self.errors[idx.item()]['z'].append([self.step_counter, self.extrinsics_trans[idx.item()][2].item() - self.ext_init[idx.item()][2].item()])
                
                    
            # === periodically render & save ===
            if self.step_counter % 1000 == 0 or self.step_counter == 1:
                fig, axes = plt.subplots(nrows=6, ncols=3, figsize=(15, 20))
                axes = axes.reshape(6, 3)  # ensure it's a 2D array even if 6x3

                colors = ['red', 'green', 'blue']
                axes_labels = ['x', 'y', 'z']
                for row, keyword in enumerate(self.errors.keys()):
                    for col, axis in enumerate(axes_labels):
                        if len(self.errors[keyword][axis]) > 0:
                            data = np.array(self.errors[keyword][axis])
                            ax = axes[row, col]
                            ax.scatter(data[:, 0], data[:, 1], color=colors[col], s=8)
                            ax.set_title(f"{keyword} - {axis}", fontsize=10)
                            ax.set_xlabel("Step")
                            ax.set_ylabel("Error Value")
                        else:
                            axes[row, col].axis('off')  # hide if no data

                plt.tight_layout()
                plt.savefig(f"calib_results/save/adjustments/adjustments_{self.step_counter}_new.png", dpi=200)
                plt.close(fig)
                for k in self.errors.keys():
                    self.errors[k] = {'x': [], 'y': [], 'z': []}

            return torch.cat(outputs, dim=0)
        else:
            raise ValueError(f"Not implemented for {self.config.mode}")
            

    def apply_to_raybundle(self, raybundle: RayBundle) -> None:
        """Apply the pose correction to the raybundle"""
        if self.config.mode != "off":
            with torch.cuda.amp.autocast(enabled=False):
                sensor_indices = raybundle.camera_indices.squeeze()
                mask_ext = sensor_indices < (self.sequence_length * self.camera_count)
                camera_to_worlds = raybundle.metadata["s2w"][mask_ext]
                unique_indices = torch.unique(sensor_indices[mask_ext])
                for idx in unique_indices:
                    with torch.no_grad():
                        pose = camera_to_worlds[sensor_indices[mask_ext] == idx][0]
                        self.camera_to_worlds[idx.item()] = pose.detach()
                
                correction_matrices = self(raybundle.camera_indices.squeeze())
                raybundle.origins = raybundle.origins + correction_matrices[:, :3, 3]
                raybundle.directions = (
                    torch.bmm(correction_matrices[:, :3, :3], raybundle.directions[..., None])
                    .squeeze()
                    .to(raybundle.origins)
                )

    def _get_ext_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return torch.cat([self.extrinsics_trans, self.extrinsics_rot], dim=-1)
    
    def _get_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return self.adjustment - self.mask_adj_rot
    
    def _get_offsets(self) -> Float[Tensor, "num_cameras 1"]:
        """Get the offsets."""
        return self.offsets

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off": 
            pose_adjustment = self._get_adjustment()
            # Add a regularization term for camera optimizer that penalizes translation and rotation adjustments.
            # loss_dict["camera_opt_regularizer"] = (
            #     pose_adjustment[:, :3, 3].mean().norm(dim=-1) * self.config.trans_l2_penalty
            #     + pose_adjustment[:, :3, :3].mean().norm(dim=-1) * self.config.rot_l2_penalty
            # )
            
            # # Increasing regularization over time
            # if self.step_counter < 1000:
            #     trans_weight = 1
            #     rot_weight = 1
            # else:
            #     # Increase regularization to prevent oscillations
            #     trans_weight = 0.03
            #     rot_weight = 0.1
                
            # loss_dict["camera_opt_regularizer"] = (
            #     pose_adjustment[:, :3, 3].mean().norm(dim=-1) * trans_weight
            #     + pose_adjustment[:, :3, :3].mean().norm(dim=-1) * rot_weight
            # )

    def get_metrics_dict(self, metrics_dict: dict) -> None:
        """Get camera optimizer metrics"""
        if self.config.mode != "off":
            pose_adjustment = self._get_adjustment()
            metrics_dict["camera_opt_translation"] = pose_adjustment[:, :3, 3].norm(dim=-1).mean()
            metrics_dict["camera_opt_rotation"] = pose_adjustment[:, :3, :3].norm(dim=-1).mean()
            # metrics_dict["camera_opt_offset"] = self.offsets.norm()

    def get_param_groups(self, param_groups: dict) -> None:
        """Get camera optimizer parameters"""
        if self.config.mode != "off":
            param_groups["camera_opt_trans"] = [self.extrinsics_trans]
            param_groups["camera_opt_rot"]   = [self.extrinsics_rot]
        else:
            assert len(camera_opt_params) == 0


@dataclass
class ScaledCameraOptimizerConfig(CameraOptimizerConfig):
    """Configuration of axis-masked optimization for camera poses."""

    _target: Type = field(default_factory=lambda: ScaledCameraOptimizer)

    weights: Tuple[float, float, float, float, float, float] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )

    trans_l2_penalty: Union[Tuple[float, float, float], float] = (
        1e-2,
        1e-2,
        1e-2,
    )  # TODO: this is l1


class ScaledCameraOptimizer(CameraOptimizer):
    """Camera optimizer that masks which components can be optimized."""

    def __init__(self, config: ScaledCameraOptimizerConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.config: ScaledCameraOptimizerConfig = self.config
        self.register_buffer("weights", torch.tensor(self.config.weights, dtype=torch.float32))
        self.trans_penalty = torch.tensor(self.config.trans_l2_penalty, dtype=torch.float32, device=self.device)

    def _get_pose_adjustment(self) -> Float[Tensor, "num_cameras 6"]:
        """Get the pose adjustment."""
        return self.pose_adjustment * self.weights

    def get_loss_dict(self, loss_dict: dict) -> None:
        """Add regularization"""
        if self.config.mode != "off":
            pose_adjustment = self._get_pose_adjustment()
            self.trans_penalty = self.trans_penalty.to(pose_adjustment.device)
            loss_dict["camera_opt_regularizer"] = (
                pose_adjustment[:, :3].abs() * self.trans_penalty
            ).mean() + pose_adjustment[:, 3:].norm(dim=-1).mean() * self.config.rot_l2_penalty
