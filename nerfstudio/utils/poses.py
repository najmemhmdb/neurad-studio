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
Common 3D pose methods
"""

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor
from scipy.spatial.transform import Slerp
from nerfstudio.cameras.camera_utils import rotmat_to_unitquat, unitquat_slerp, unitquat_slerp_fast, unitquat_to_rotmat
from scipy.spatial.transform import Rotation as R

def to4x4(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 4 4"]:
    """Convert 3x4 pose matrices to a 4x4 with the addition of a homogeneous coordinate.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Camera poses with additional homogenous coordinate added.
    """
    constants = torch.zeros_like(pose[..., :1, :], device=pose.device)
    constants[..., :, 3] = 1
    return torch.cat([pose, constants], dim=-2)


def inverse(pose: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Invert provided pose matrix.

    Args:
        pose: Camera pose without homogenous coordinate.

    Returns:
        Inverse of pose.
    """
    R = pose[..., :3, :3]
    t = pose[..., :3, 3:]
    R_inverse = R.transpose(-2, -1)
    t_inverse = -R_inverse.matmul(t)
    return torch.cat([R_inverse, t_inverse], dim=-1)


def multiply(pose_a: Float[Tensor, "*batch 3 4"], pose_b: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Multiply two pose matrices, A @ B.

    Args:
        pose_a: Left pose matrix, usually a transformation applied to the right.
        pose_b: Right pose matrix, usually a camera pose that will be transformed by pose_a.

    Returns:
        Camera pose matrix where pose_a was applied to pose_b.
    """
    R1, t1 = pose_a[..., :3, :3], pose_a[..., :3, 3:]
    R2, t2 = pose_b[..., :3, :3], pose_b[..., :3, 3:]
    R = R1.matmul(R2)
    t = t1 + R1.matmul(t2)
    return torch.cat([R, t], dim=-1)


def normalize(poses: Float[Tensor, "*batch 3 4"]) -> Float[Tensor, "*batch 3 4"]:
    """Normalize the XYZs of poses to fit within a unit cube ([-1, 1]). Note: This operation is not in-place.

    Args:
        poses: A collection of poses to be normalized.

    Returns:
        Normalized collection of poses.
    """
    pose_copy = torch.clone(poses)
    pose_copy[..., :3, 3] /= torch.max(torch.abs(poses[..., :3, 3]))

    return pose_copy


def interpolate_trajectories_6d(poses, pose_times, query_times, pose_valid_mask=None, flatten=True):
    """Interpolate trajectory poses at query times using linear interpolation.

    Args:
        poses: A collection of 9d representation poses (6d rot + 3d position) to be interpolated N_times x N_trajectories.
        pose_times: The timestamps of the poses N_times.
        query_times: The timestamps to interpolate the poses at [N_queries, 1].
        pose_valid_mask: A mask indicating which poses are valid N_times x N_trajectories.
        flatten: Whether to return the interpolated poses as a flat (masked) tensor or as a 3D (unmasked) tensor. Flatten also returns the indices of the queries and objects used for interpolation. Not flattening returns the mask used to find said indices. Default is True.

    Returns:
        The interpolated poses at the query times (M x 9)
        The indices of the queries used for interpolation (M).
        The indices of the tractories used for interpolation (M).
    """
    assert len(poses.shape) == 3, "Poses must be of shape [num_actors, num_poses, 9]"
    if len(poses) == 0:
        if flatten:
            return (
                torch.empty(0, 9, device=poses.device),
                torch.empty(0, dtype=torch.long),
                torch.empty(0, dtype=torch.long),
            )
        else:
            return torch.empty(0, 9, device=poses.device), torch.empty(len(query_times), 0, dtype=torch.bool)
    # Orthogonalize the first two axes
    a1 = F.normalize(poses[..., :3], dim=-1)
    a2 = poses[..., 3:6]
    a2 = a2 - (a1 * a2).sum(-1, keepdim=True) * a1
    a2 = F.normalize(a2, dim=-1)

    positions = poses[..., 6:9]
    poses = torch.cat([a1, a2, positions], dim=-1)
    query_times = query_times.squeeze(-1)
    right_idx = torch.searchsorted(pose_times, query_times)
    left_idx = (right_idx - 1).clamp(min=0)
    right_idx = right_idx.clamp(max=len(pose_times) - 1)

    # Compute the fraction between the left (previous) and right (after) timestamps
    right_time = pose_times[right_idx]
    left_time = pose_times[left_idx]
    time_diff = right_time - left_time + 1e-6
    fraction = (query_times - left_time) / time_diff  # 0 = all left, 1 = all right
    fraction = fraction.clamp(0.0, 1.0)  # clamp to handle out of bounds

    if pose_valid_mask is None:
        pose_valid_mask = torch.ones_like(poses[..., 0, 0], dtype=torch.bool)
    trajs_to_sample = pose_valid_mask[left_idx] | pose_valid_mask[right_idx]  # [num_queries, n_trajs]

    poses_left = poses[left_idx]
    poses_right = poses[right_idx]

    if flatten:
        query_idxs, object_idxs = torch.where(trajs_to_sample)  # 2 x [num_queries*n_valid_trajs]
        poses_left = poses_left[query_idxs, object_idxs]  # [num_queries*n_valid_objects, 9]
        poses_right = poses_right[query_idxs, object_idxs]  # [num_queries*n_valid_objects, 9]
        interpolated_poses = poses_left + (poses_right - poses_left) * fraction[query_idxs].unsqueeze(-1)
        return interpolated_poses, query_idxs, object_idxs
    else:
        interpolated_poses = poses_left + (poses_right - poses_left) * fraction.unsqueeze(-1).unsqueeze(-1)
        return interpolated_poses, trajs_to_sample  # trajs_to_sample is a validity mask


def interpolate_trajectories(poses, pose_times, query_times, pose_valid_mask=None, clamp_frac=True):
    """Interpolate trajectory poses at query times using linear interpolation.

    Args:
        poses: A collection of poses to be interpolated N_times x N_trajectories.
        pose_times: The timestamps of the poses N_times.
        query_times: The timestamps to interpolate the poses at N_queries.
        pose_valid_mask: A mask indicating which poses are valid N_times x N_trajectories.

    Returns:
        The interpolated poses at the query times (M x 3 x 4)
        The indices of the queries used for interpolation (M).
        The indices of the tractories used for interpolation (M).
    """
    assert len(poses.shape) == 4, "Poses must be of shape [num_poses, num_actors, 3, 4]"
    # bugs are crawling like crazy if we only have one query time, fix with maybe squeeze
    qt = query_times if query_times.shape[-1] == 1 else query_times.squeeze()
    right_idx = torch.searchsorted(pose_times, qt)
    right_idx = right_idx.clamp(min=1, max=len(pose_times) - 1)
    left_idx = right_idx - 1

    # Compute the fraction between the left (previous) and right (after) timestamps
    right_time = pose_times[right_idx]
    left_time = pose_times[left_idx]
    time_diff = right_time - left_time + 1e-6
    fraction = (qt - left_time) / time_diff  # 0 = all left, 1 = all right
    if clamp_frac:
        fraction = fraction.clamp(0.0, 1.0)  # clamp to handle out of bounds

    if pose_valid_mask is None:
        pose_valid_mask = torch.ones_like(poses[..., 0, 0], dtype=torch.bool)
    trajs_to_sample = pose_valid_mask[left_idx] | pose_valid_mask[right_idx]  # [num_queries, n_trajs]
    query_idxs, object_idxs = torch.where(trajs_to_sample)  # 2 x [num_queries*n_valid_trajs]
    quat = rotmat_to_unitquat(poses[..., :3, :3])
    quat_left = quat[left_idx][query_idxs, object_idxs]  # [num_queries*n_valid_objects, 4]
    quat_right = quat[right_idx][query_idxs, object_idxs]  # [num_queries*n_valid_objects, 4]
    interp_fn = unitquat_slerp if (fraction < 0).any() or (fraction > 1).any() else unitquat_slerp_fast
    interp_quat = interp_fn(quat_left, quat_right, fraction[query_idxs])
    interp_rot = unitquat_to_rotmat(interp_quat)

    pos_left = poses[left_idx][query_idxs, object_idxs, :3, 3]  # [num_queries*n_valid_objects, 3]
    pos_right = poses[right_idx][query_idxs, object_idxs, :3, 3]  # [num_queries*n_valid_objects, 3]
    interp_pos = pos_left + (pos_right - pos_left) * fraction[query_idxs].unsqueeze(-1)

    interpolated_poses = torch.cat([interp_rot, interp_pos.unsqueeze(-1)], dim=-1)
    return interpolated_poses, query_idxs, object_idxs

def interpolate_velocities(velocities, pose_times, query_times, clamp_frac=False):
    qt = query_times if query_times.shape[-1] == 1 else query_times.view(-1)
    right_idx = torch.searchsorted(pose_times, qt)
    right_idx = right_idx.clamp(min=1, max=len(pose_times) - 1)
    left_idx = right_idx - 1

    # Compute the fraction between the left (previous) and right (after) timestamps
    right_time = pose_times[right_idx]
    left_time = pose_times[left_idx]
    time_diff = right_time - left_time + 1e-6
    fraction = (qt - left_time) / time_diff  # 0 = all left, 1 = all right
    if clamp_frac:
        fraction = fraction.clamp(0.0, 1.0)  # clamp to handle out of bounds
    velocity_left = velocities[left_idx]
    velocity_right = velocities[right_idx]
    interpolated_velocities = velocity_left + (velocity_right - velocity_left) * fraction.unsqueeze(-1)
    return interpolated_velocities

def rotation_difference(rot1, rot2):
    """Compute the difference between two rotations, i.e., how to rotate rot1 to get to rot2.
       Equivalent to "rot2 - rot1".

    Args:
        rot1: A rotation matrix [..., 3, 3].
        rot2: A rotation matrix [..., 3, 3].

    Returns:
        The difference between the two rotations, expressed as axis-angle representation.
    """

    # Relative rotation matrix
    R_rel = rot1.transpose(-2, -1) @ rot2
    # Compute the angle of rotation
    theta = torch.acos(((R_rel.diagonal(dim1=-2, dim2=-1).sum(dim=-1) - 1) / 2).clamp(-1, 1))

    # Compute the axis of rotation
    axis = torch.stack(
        [
            R_rel[..., 2, 1] - R_rel[..., 1, 2],
            R_rel[..., 0, 2] - R_rel[..., 2, 0],
            R_rel[..., 1, 0] - R_rel[..., 0, 1],
        ],
        dim=-1,
    )
    axis = F.normalize(axis, dim=-1)

    return theta.unsqueeze(-1) * axis

def vectorized_interpolate(poses, pose_times, query_times):
    """
    Interpolate SE(3) poses at arbitrary query_times.
    poses:      [N, 3, 4]   (world_T_lidar at pose_times)
    pose_times: [N, 1] or [N] (monotonic, same units as query_times)
    query_times:[B, 1] or [B]
    returns:    [B, 3, 4]
    """
    device = poses.device
    dtype  = poses.dtype

    # Ensure shapes
    pose_times  = pose_times.view(-1)
    query_times = query_times.view(-1)

    N = pose_times.shape[0]
    B = query_times.shape[0]

    # 1) find the index of the first pose time strictly greater than each query
    #    i1 in [0..N], clamp to [1..N-1] to ensure we have both neighbors
    i1 = torch.searchsorted(pose_times, query_times, right=False)
    i1 = torch.clamp(i1, 1, N - 1)
    i0 = i1 - 1

    t0 = pose_times[i0]
    t1 = pose_times[i1]
    # Avoid divide-by-zero if duplicate stamps
    denom = (t1 - t0).clamp_min(1e-9)
    alpha = ((query_times - t0) / denom).to(dtype)  # [B]

    # 2) gather neighbor poses
    P0 = poses[i0]  # [B,3,4]
    P1 = poses[i1]  # [B,3,4]

    R0 = P0[..., :3, :3]
    t0p = P0[..., :3, 3]
    R1 = P1[..., :3, :3]
    t1p = P1[..., :3, 3]

    # 3) rotation: SLERP
    q0 = rotmat_to_unitquat(R0)                 # [B,4]
    q1 = rotmat_to_unitquat(R1)                 # [B,4]
    q  = unitquat_slerp_fast(q0, q1, alpha)     # [B,4]
    R  = unitquat_to_rotmat(q)                  # [B,3,3]

    # 4) translation: linear in world frame
    t  = t0p + alpha.unsqueeze(-1) * (t1p - t0p)  # [B,3]
    # t  = t0p 
    out = torch.zeros(B, 3, 4, dtype=dtype, device=device)
    out[..., :3, :3] = R
    out[..., :3, 3]  = t
    return out