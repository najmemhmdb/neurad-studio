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
Helper for Lie group operations. Currently only used for pose optimization.
"""
import torch
from jaxtyping import Float
from torch import Tensor


# We make an exception on snake case conventions because SO3 != so3.
def exp_map_SO3xR3(tangent_vector: Float[Tensor, "b 6"]) -> Float[Tensor, "b 3 4"]:
    """Compute the exponential map of the direct product group `SO(3) x R^3`.

    This can be used for learning pose deltas on SE(3), and is generally faster than `exp_map_SE3`.

    Args:
        tangent_vector: Tangent vector; length-3 translations, followed by an `so(3)` tangent vector.
    Returns:
        [R|t] transformation matrices.
    """
    # code for SO3 map grabbed from pytorch3d and stripped down to bare-bones
    log_rot = tangent_vector[:, 3:]
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, 1e-4).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = torch.zeros((log_rot.shape[0], 3, 3), dtype=log_rot.dtype, device=log_rot.device)
    skews[:, 0, 1] = -log_rot[:, 2]
    skews[:, 0, 2] = log_rot[:, 1]
    skews[:, 1, 0] = log_rot[:, 2]
    skews[:, 1, 2] = -log_rot[:, 0]
    skews[:, 2, 0] = -log_rot[:, 1]
    skews[:, 2, 1] = log_rot[:, 0]
    skews_square = torch.bmm(skews, skews)

    ret = torch.zeros(tangent_vector.shape[0], 3, 4, dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = (
        fac1[:, None, None] * skews
        + fac2[:, None, None] * skews_square
        + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None]
    )

    # Compute the translation
    ret[:, :3, 3] = tangent_vector[:, :3]
    return ret


def exp_map_SE3(tangent_vector: Float[Tensor, "b 6"]) -> Float[Tensor, "b 3 4"]:
    """Compute the exponential map `se(3) -> SE(3)`.

    This can be used for learning pose deltas on `SE(3)`.

    Args:
        tangent_vector: A tangent vector from `se(3)`.

    Returns:
        [R|t] transformation matrices.
    """

    tangent_vector_lin = tangent_vector[:, :3].view(-1, 3, 1)
    tangent_vector_ang = tangent_vector[:, 3:].view(-1, 3, 1)

    theta = torch.linalg.norm(tangent_vector_ang, dim=1).unsqueeze(1)
    theta2 = theta**2
    theta3 = theta**3

    near_zero = theta < 1e-2
    non_zero = torch.ones(1, dtype=tangent_vector.dtype, device=tangent_vector.device)
    theta_nz = torch.where(near_zero, non_zero, theta)
    theta2_nz = torch.where(near_zero, non_zero, theta2)
    theta3_nz = torch.where(near_zero, non_zero, theta3)

    # Compute the rotation
    sine = theta.sin()
    cosine = torch.where(near_zero, 8 / (4 + theta2) - 1, theta.cos())
    sine_by_theta = torch.where(near_zero, 0.5 * cosine + 0.5, sine / theta_nz)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 * sine_by_theta, (1 - cosine) / theta2_nz)
    ret = torch.zeros(tangent_vector.shape[0], 3, 4).to(dtype=tangent_vector.dtype, device=tangent_vector.device)
    ret[:, :3, :3] = one_minus_cosine_by_theta2 * tangent_vector_ang @ tangent_vector_ang.transpose(1, 2)

    ret[:, 0, 0] += cosine.view(-1)
    ret[:, 1, 1] += cosine.view(-1)
    ret[:, 2, 2] += cosine.view(-1)
    temp = sine_by_theta.view(-1, 1) * tangent_vector_ang.view(-1, 3)
    ret[:, 0, 1] -= temp[:, 2]
    ret[:, 1, 0] += temp[:, 2]
    ret[:, 0, 2] += temp[:, 1]
    ret[:, 2, 0] -= temp[:, 1]
    ret[:, 1, 2] -= temp[:, 0]
    ret[:, 2, 1] += temp[:, 0]

    # Compute the translation
    sine_by_theta = torch.where(near_zero, 1 - theta2 / 6, sine_by_theta)
    one_minus_cosine_by_theta2 = torch.where(near_zero, 0.5 - theta2 / 24, one_minus_cosine_by_theta2)
    theta_minus_sine_by_theta3_t = torch.where(near_zero, 1.0 / 6 - theta2 / 120, (theta - sine) / theta3_nz)

    ret[:, :, 3:] = sine_by_theta * tangent_vector_lin
    ret[:, :, 3:] += one_minus_cosine_by_theta2 * torch.cross(tangent_vector_ang, tangent_vector_lin, dim=1)
    ret[:, :, 3:] += theta_minus_sine_by_theta3_t * (
        tangent_vector_ang @ (tangent_vector_ang.transpose(1, 2) @ tangent_vector_lin)
    )
    return ret


import torch
from jaxtyping import Float
from torch import Tensor

def log_map_SE3(transform: Float[Tensor, "b 3 4"]) -> Float[Tensor, "b 6"]:
    """Compute the logarithm map `SE(3) -> se(3)`.
    
    This is the inverse of exp_map_SE3, converting transformation matrices
    back to tangent vectors.
    
    Args:
        transform: [R|t] transformation matrices of shape (b, 3, 4).
    
    Returns:
        tangent_vector: A tangent vector from se(3) of shape (b, 6).
            [:, :3] = linear part, [:, 3:] = angular part
    """
    batch_size = transform.shape[0]
    device = transform.device
    dtype = transform.dtype
    
    # Extract rotation and translation
    R = transform[:, :3, :3]  # (b, 3, 3)
    t = transform[:, :3, 3]  # (b, 3)
    
    # Compute rotation angle
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    cos_theta = (trace - 1.0) / 2.0
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
    
    # Initialize output
    tangent_vector = torch.zeros(batch_size, 6, dtype=dtype, device=device)
    
    # Determine which case we're in
    near_zero = theta < 1e-2
    near_pi = theta > (torch.pi - 1e-2)
    normal = ~(near_zero | near_pi)
    
    # NEAR ZERO CASE
    if near_zero.any():
        idx = near_zero
        # omega from antisymmetric part
        tangent_vector[idx, 3] = (R[idx, 2, 1] - R[idx, 1, 2]) / 2.0
        tangent_vector[idx, 4] = (R[idx, 0, 2] - R[idx, 2, 0]) / 2.0
        tangent_vector[idx, 5] = (R[idx, 1, 0] - R[idx, 0, 1]) / 2.0
        # translation is just t
        tangent_vector[idx, :3] = t[idx]
    
    # NEAR PI CASE (θ ≈ π)
    if near_pi.any():
        idx = near_pi
        n = idx.sum().item()
        
        for i in range(n):
            batch_idx = torch.where(idx)[0][i]
            R_i = R[batch_idx]
            theta_i = theta[batch_idx]
            
            # Find the rotation axis from (R + I)
            # The axis is the eigenvector corresponding to eigenvalue +1
            B = R_i + torch.eye(3, dtype=dtype, device=device)
            
            # Find column with largest norm (most stable)
            col_norms = torch.sqrt((B ** 2).sum(dim=0))
            max_col = torch.argmax(col_norms)
            
            # Extract and normalize the axis
            k = B[:, max_col]
            k = k / torch.linalg.norm(k)
            
            # Determine the sign of k using the antisymmetric part
            # For theta near pi, R - R^T is still informative about direction
            R_antisym = R_i - R_i.T
            # Extract the direction from antisymmetric part
            v_antisym = torch.tensor([R_antisym[2, 1], R_antisym[0, 2], R_antisym[1, 0]], 
                                     dtype=dtype, device=device)
            
            # If the signs don't match, flip k
            if torch.dot(k, v_antisym) < 0:
                k = -k
            
            # omega = theta * k
            omega = theta_i * k
            tangent_vector[batch_idx, 3:] = omega
            
            # Compute V_inv for translation
            omega_col = omega.view(3, 1)
            
            # Build skew-symmetric matrix
            omega_skew = torch.zeros(3, 3, dtype=dtype, device=device)
            omega_skew[0, 1] = -omega[2]
            omega_skew[0, 2] = omega[1]
            omega_skew[1, 0] = omega[2]
            omega_skew[1, 2] = -omega[0]
            omega_skew[2, 0] = -omega[1]
            omega_skew[2, 1] = omega[0]
            
            omega_skew_sq = omega_skew @ omega_skew
            theta2 = theta_i ** 2
            half_theta = theta_i / 2.0
            
            # For theta near pi, use the limiting form
            # V^{-1} = I - (1/2)*[omega]_x + ((1 - (theta/2)*cot(theta/2))/theta^2)*[omega]_x^2
            sin_half = torch.sin(half_theta)
            cos_half = torch.cos(half_theta)
            
            if sin_half.abs() > 1e-6:
                cot_half = cos_half / sin_half
                coeff = (1.0 - half_theta * cot_half) / theta2
            else:
                # Use Taylor expansion for theta very close to pi
                coeff = 1.0 / 12.0  # limiting value
            
            V_inv = (torch.eye(3, dtype=dtype, device=device)
                     - 0.5 * omega_skew
                     + coeff * omega_skew_sq)
            
            tangent_vector[batch_idx, :3] = (V_inv @ t[batch_idx].unsqueeze(-1)).squeeze(-1)
    
    # NORMAL CASE
    if normal.any():
        idx = normal
        theta_val = theta[idx]
        sin_theta = torch.sin(theta_val)
        
        # Extract omega (angular part)
        factor = theta_val / (2.0 * sin_theta)
        tangent_vector[idx, 3] = factor * (R[idx, 2, 1] - R[idx, 1, 2])
        tangent_vector[idx, 4] = factor * (R[idx, 0, 2] - R[idx, 2, 0])
        tangent_vector[idx, 5] = factor * (R[idx, 1, 0] - R[idx, 0, 1])
        
        # Compute V^{-1}
        omega = tangent_vector[idx, 3:].view(-1, 3, 1)  # (n, 3, 1)
        theta_val = theta_val.view(-1, 1, 1)  # (n, 1, 1)
        
        # Build skew-symmetric matrix [omega]_x
        n = idx.sum()
        omega_skew = torch.zeros(n, 3, 3, dtype=dtype, device=device)
        omega_skew[:, 0, 1] = -omega[:, 2, 0]
        omega_skew[:, 0, 2] = omega[:, 1, 0]
        omega_skew[:, 1, 0] = omega[:, 2, 0]
        omega_skew[:, 1, 2] = -omega[:, 0, 0]
        omega_skew[:, 2, 0] = -omega[:, 1, 0]
        omega_skew[:, 2, 1] = omega[:, 0, 0]
        
        omega_skew_sq = omega_skew @ omega_skew
        
        theta2 = theta_val ** 2
        half_theta = theta_val / 2.0
        
        # Coefficient for [omega]_x^2 term
        coeff = (1.0 / theta2) * (1.0 - theta_val * torch.cos(half_theta) / (2.0 * torch.sin(half_theta)))
        
        V_inv = (torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
                 - 0.5 * omega_skew
                 + coeff * omega_skew_sq)
        
        # Apply V^{-1} to translation
        t_inv = (V_inv @ t[idx].unsqueeze(-1)).squeeze(-1)
        tangent_vector[idx, :3] = t_inv
    
    return tangent_vector


def mat4_to_SO3xR3_twist(T4x4: torch.Tensor) -> torch.Tensor:
    """
    Convert a 4x4 transform to the 6D vector for exp_map_SO3xR3.
    Returns xi = [t_x, t_y, t_z, phi_x, phi_y, phi_z] (shape [6]).
    """
    T4x4 = T4x4.to(dtype=torch.float64)
    R = T4x4[:3, :3]
    t = T4x4[:3, 3]
    
    # Project R to SO(3) if needed (important for numerical stability)
    U, _, Vh = torch.linalg.svd(R)
    R_proj = U @ Vh
    if torch.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vh
    
    # Use standard log map consistent with exp_map_SO3xR3
    phi = _rotmat_to_rotvec_consistent(R_proj)

    return torch.cat([t, phi])

def _rotmat_to_rotvec_consistent(R: torch.Tensor) -> torch.Tensor:
    trace = torch.clamp(torch.trace(R), -1.0, 3.0)
    cos_theta = (trace - 1.0) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    eps_small = 1e-6
    eps_pi = 1e-4  # tolerance around pi

    if theta < eps_small:
        # Small-angle approximation
        rx = 0.5 * (R[2, 1] - R[1, 2])
        ry = 0.5 * (R[0, 2] - R[2, 0])
        rz = 0.5 * (R[1, 0] - R[0, 1])
        phi = torch.stack([rx, ry, rz])

    elif abs(theta - torch.pi) < eps_pi:
        # Special handling for rotations close to 180 degrees
        # See e.g. Shoemake / Graphics Gems or classic SO(3) log map implementations
        R_diag = torch.diagonal(R)
        axis_sq = (R_diag + 1.0) * 0.5  # components of axis^2

        # Numerical safety
        axis_sq = torch.clamp(axis_sq, min=0.0)

        # Pick the largest diagonal entry to avoid division by small numbers
        if axis_sq[0] >= axis_sq[1] and axis_sq[0] >= axis_sq[2]:
            x = torch.sqrt(axis_sq[0])
            y = R[0, 1] / (2.0 * x) if x > eps_small else 0.0
            z = R[0, 2] / (2.0 * x) if x > eps_small else 0.0
            axis = torch.stack([x, y, z])
        elif axis_sq[1] >= axis_sq[0] and axis_sq[1] >= axis_sq[2]:
            y = torch.sqrt(axis_sq[1])
            x = R[0, 1] / (2.0 * y) if y > eps_small else 0.0
            z = R[1, 2] / (2.0 * y) if y > eps_small else 0.0
            axis = torch.stack([x, y, z])
        else:
            z = torch.sqrt(axis_sq[2])
            x = R[0, 2] / (2.0 * z) if z > eps_small else 0.0
            y = R[1, 2] / (2.0 * z) if z > eps_small else 0.0
            axis = torch.stack([x, y, z])

        axis = axis / axis.norm()
        phi = axis * theta

    else:
        # Standard case: use antisymmetric part
        axis = torch.stack([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (2.0 * torch.sin(theta))
        phi = axis * theta

    # Optional consistency check with exp map, and sign flip if needed:
    test_twist = torch.cat([torch.zeros(3, dtype=R.dtype, device=R.device), phi]).unsqueeze(0)
    R_recon = exp_map_SO3xR3(test_twist)[0, :3, :3]

    if torch.norm(R_recon - R) > 1e-4:
        phi_neg = -phi
        test_twist_neg = torch.cat(
            [torch.zeros(3, dtype=R.dtype, device=R.device), phi_neg]
        ).unsqueeze(0)
        R_recon_neg = exp_map_SO3xR3(test_twist_neg)[0, :3, :3]
        if torch.norm(R_recon_neg - R) < torch.norm(R_recon - R):
            phi = phi_neg

    return phi
