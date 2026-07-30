import math
import torch
import torch.nn.functional as F

def simulate_events(
    frames: torch.Tensor,
    threshold: float = 0.1,
    eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        frames - floats (0 to 1) of shape [B, T, C, H, W]
        threshold - contrast sensitivity
        eps - prevent log(0)
    Returns:
        dense events - [B, T-1, H, W] in (-1, 0, 1)
        sparse_coords - [N, 5] triggered events, ie
            batch, time, y, x, polarity.     
    """
    device = frames.device
    dtype = frames.dtype

    if frames.ndim == 5 and frames.shape[2] == 3:
        gray_frames = (
            0.2989 * frames[:, :, 0] + 
            0.5870 * frames[:, :, 1] + 
            0.1140 * frames[:, :, 2]            
        ).unsqueeze(2)
    else:
        gray_frames = frames

    log_intensity = torch.log(gray_frames + eps, dtype=dtype, device=device)
    diffs = (log_intensity[:, 1:] - log_intensity[:, :-1]).squeeze(2)

    pos_events = (diffs >= threshold).to(torch.int8)
    neg_events = (diffs <= -threshold).to(torch.int8)
    dense_events = pos_events - neg_events

    active_indices = torch.nonzero(dense_events) # [N, 4]
                                                 # (b, t, y, x)
    polarities = dense_events[
        active_indices[:, 0],
        active_indices[:, 1],
        active_indices[:, 2],
        active_indices[:, 3]
    ]

    sparse_coords = torch.cat(
        [active_indices, polarities.unsqueeze(-1).to(torch.long)],
        dim=-1
    )
    return dense_events, sparse_coords

def isotropic(
    sparse_coords: torch.Tensor,
    spatial_dims: tuple[int, int],
    a: float = 0.05,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
        maps coords to isotropic manifold - schwartz log-polar
        conformal transform.
        Args:
            sparse_coords - [N, 5] (batch, time, y, x, polarity)
            spatial_dims - [H, W] of visual field
            a - foveal compression factor
            scale - output scaling
        Return:
            cortical_coords - tensor of float on the hemisphere
            hemisphere_idx - 0 for right, 1 for left visual field
    """
    H, W = spatial_dims
    y = sparse_coords[:, 2].to(torch.float32)
    x = sparse_coords[:, 3].to(torch.float32)

    # center and normalize to -1, 1
    y_c = (y - (H - 1) / 2.0) / (H / 2.0)
    x_c = (x - (W - 1) / 2.0) / (W / 2.0)

    x_hemi = torch.abs(x_c)
    hemisphere_idx = (x_c < 0).to(torch.long)

    u = 0.5 * torch.log(
        ((x_hemi + a) ** 2 + y_c ** 2) / a ** 2
    )
    v = torch.atan2(y_c, x_hemi + a)

    iso_coords = torch.stack([u, v], dim=-1) * scale
    return iso_coords, hemisphere_idx

def warp_to_isotropic(
    frames: torch.Tensor,
    cortical_dims: tuple[int, int] = (64, 64),
    a: float = 0.05,
 ) -> tuple[torch.Tensor, torch.Tensor]:
    """
        directly warps dense frames.
    """
    B, C, H, W = frames.shape
    Hc, Wc = cortical_dims
    device = frames.device
    dtype = frames.dtype
    
    # determine the max u boundary (where visual boundary radius is 1.0)
    u_max = math.log(1.0 + 1.0 / a)
    v_max = math.pi / 2.0  # Captures a 180 visual hemi-field
    
    # generate the destination grid mesh
    u_vals = torch.linspace(0.0, u_max, Wc, device=device, dtype=dtype)
    v_vals = torch.linspace(-v_max, v_max, Hc, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v_vals, u_vals, indexing="ij")
    
    # inverse map: z = a * (exp(w) - 1)
    x_hemi = a * (torch.exp(u_grid) * torch.cos(v_grid) - 1.0)
    y_c = a * torch.exp(u_grid) * torch.sin(v_grid)
    
    # expand for batch operations
    grid_right_vis = torch.stack([x_hemi, y_c], dim=-1).expand(B, Hc, Wc, 2)
    grid_left_vis = torch.stack([-x_hemi, y_c], dim=-1).expand(B, Hc, Wc, 2)
    
    # bilinear sampling with zero-padding outside visual boundaries
    left_cortical_hemi = F.grid_sample(frames, grid_right_vis, mode="bilinear", padding_mode="zeros", align_corners=True)
    right_cortical_hemi = F.grid_sample(frames, grid_left_vis, mode="bilinear", padding_mode="zeros", align_corners=True)
    
    return left_cortical_hemi, right_cortical_hemi
    
