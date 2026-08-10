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
    if frames.ndim == 5 and frames.shape[2] == 3:
        gray_frames = (
            0.2989 * frames[:, :, 0]
            + 0.5870 * frames[:, :, 1]
            + 0.1140 * frames[:, :, 2]
        ).unsqueeze(2)
    else:
        gray_frames = frames

    log_intensity = torch.log(gray_frames + eps)
    diffs = (log_intensity[:, 1:] - log_intensity[:, :-1]).squeeze(2)

    pos_events = (diffs >= threshold).to(torch.int8)
    neg_events = (diffs <= -threshold).to(torch.int8)
    dense_events = pos_events - neg_events

    active_indices = torch.nonzero(dense_events)  # [N, 4] (b, t, y, x)
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
    Maps coords to isotropic manifold — Schwartz log-polar conformal transform.

    Args:
        sparse_coords - [N, 5] (batch, time, y, x, polarity)
        spatial_dims - [H, W] of visual field
        a - foveal compression factor
        scale - output scaling

    Returns:
        isotropic_coords - tensor of float on the hemisphere
        hemisphere_idx - 0 for right, 1 for left visual field
    """
    H, W = spatial_dims
    y = sparse_coords[:, 2].to(torch.float32)
    x = sparse_coords[:, 3].to(torch.float32)

    # center and normalize to [-1, 1]
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
    isotropic_dims: tuple[int, int] = (64, 64),
    a: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Directly warps dense frames.
    """
    B, C, H, W = frames.shape
    Hc, Wc = isotropic_dims
    device = frames.device
    dtype = frames.dtype

    u_max = math.log(1.0 + 1.0 / a)
    v_max = math.pi / 2.0

    u_vals = torch.linspace(0.0, u_max, Wc, device=device, dtype=dtype)
    v_vals = torch.linspace(-v_max, v_max, Hc, device=device, dtype=dtype)
    v_grid, u_grid = torch.meshgrid(v_vals, u_vals, indexing="ij")

    x_hemi = a * (torch.exp(u_grid) * torch.cos(v_grid) - 1.0)
    y_c = a * torch.exp(u_grid) * torch.sin(v_grid)

    grid_right_vis = torch.stack([x_hemi, y_c], dim=-1).expand(B, Hc, Wc, 2)
    grid_left_vis = torch.stack([-x_hemi, y_c], dim=-1).expand(B, Hc, Wc, 2)

    left_isotropic_hemi = F.grid_sample(
        frames, grid_right_vis, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    right_isotropic_hemi = F.grid_sample(
        frames, grid_left_vis, mode="bilinear", padding_mode="zeros", align_corners=True
    )

    return left_isotropic_hemi, right_isotropic_hemi


def render_cartesian_events(
    sparse_coords: torch.Tensor,
    frame_shape: tuple[int, int, int, int],
    background_frames: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Renders sparse Cartesian event coordinates onto a 2D image grid [B, T, 3, H, W].
    Positive events (+1) render as Cyan/Green, Negative events (-1) render as Magenta/Red.

    Args:
        sparse_coords: Long tensor [N, 5] -> (batch, time, y, x, polarity).
        frame_shape: (B, T, H, W) of the target video.
        background_frames: Optional tensor [B, T, H, W] or [B, T, C, H, W] to overlay on.

    Returns:
        canvas: Float RGB tensor [B, T, 3, H, W] in range [0, 1].
    """
    B, T, H, W = frame_shape
    device = sparse_coords.device if sparse_coords.numel() > 0 else torch.device("cpu")

    if background_frames is not None:
        bg = background_frames.detach().clone()
        if bg.ndim == 4:
            bg = bg.unsqueeze(2).repeat(1, 1, 3, 1, 1)
        elif bg.shape[2] == 1:
            bg = bg.repeat(1, 1, 3, 1, 1)
        canvas = bg * 0.4
    else:
        canvas = torch.zeros(B, T, 3, H, W, device=device)

    if sparse_coords.numel() == 0:
        return canvas

    b_idx, t_idx, y_idx, x_idx, pol = (
        sparse_coords[:, 0],
        sparse_coords[:, 1],
        sparse_coords[:, 2],
        sparse_coords[:, 3],
        sparse_coords[:, 4],
    )

    # Bound check guard
    valid = (b_idx < B) & (t_idx < T) & (y_idx < H) & (x_idx < W)
    b_idx, t_idx, y_idx, x_idx, pol = b_idx[valid], t_idx[valid], y_idx[valid], x_idx[valid], pol[valid]

    pos = pol > 0
    canvas[b_idx[pos], t_idx[pos], 1, y_idx[pos], x_idx[pos]] = 1.0
    canvas[b_idx[pos], t_idx[pos], 2, y_idx[pos], x_idx[pos]] = 1.0

    neg = pol < 0
    canvas[b_idx[neg], t_idx[neg], 0, y_idx[neg], x_idx[neg]] = 1.0

    return canvas


def render_isotropic_events(
    sparse_coords: torch.Tensor,
    isotropic_coords: torch.Tensor,
    hemisphere_idx: torch.Tensor,
    frame_shape: tuple[int, int, int, int],
    isotropic_dims: tuple[int, int] = (64, 64),
    a: float = 0.05
) -> torch.Tensor:
    """
    Renders sparse isotropic manifold coordinates onto a 2D canvas [B, T, 3, Hc, 2 * Wc].
    Left Half  -> Left V1 Hemisphere (Right Visual Field).
    Right Half -> Right V1 Hemisphere (Left Visual Field).

    Args:
        sparse_coords: Long tensor [N, 5] -> (batch, time, y, x, polarity).
        isotropic_coords: Float tensor [N, 2] -> (u, v) continuous isotropic coordinates.
        hemisphere_idx: Long tensor [N] -> 0 for Right VF (Left V1), 1 for Left VF (Right V1).
        frame_shape: (B, T, H, W) of the original video.
        isotropic_dims: (Hc, Wc) spatial resolution per isotropic hemisphere.
        a: Foveal compression factor used during Schwartz mapping.

    Returns:
        canvas: Float RGB tensor [B, T, 3, Hc, 2 * Wc] in range [0, 1].
    """
    B, T, H, W = frame_shape
    Hc, Wc = isotropic_dims
    device = sparse_coords.device if sparse_coords.numel() > 0 else torch.device("cpu")

    canvas = torch.zeros(B, T, 3, Hc, 2 * Wc, device=device)

    if sparse_coords.numel() == 0 or isotropic_coords.numel() == 0:
        return canvas

    u_max = math.log(1.0 + 1.0 / a)
    v_max = math.pi / 2.0

    u, v = isotropic_coords[:, 0], isotropic_coords[:, 1]

    x_cort = torch.clamp(torch.round((u / u_max) * (Wc - 1)).to(torch.long), 0, Wc - 1)
    y_cort = torch.clamp(torch.round(((v + v_max) / (2.0 * v_max)) * (Hc - 1)).to(torch.long), 0, Hc - 1)

    x_canvas = x_cort + hemisphere_idx * Wc

    b_idx, t_idx, pol = sparse_coords[:, 0], sparse_coords[:, 1], sparse_coords[:, 4]

    # Bound check guard
    valid = (b_idx < B) & (t_idx < T) & (y_cort < Hc) & (x_canvas < 2 * Wc)
    b_idx, t_idx, y_cort, x_canvas, pol = b_idx[valid], t_idx[valid], y_cort[valid], x_canvas[valid], pol[valid]

    pos = pol > 0
    canvas[b_idx[pos], t_idx[pos], 1, y_cort[pos], x_canvas[pos]] = 1.0
    canvas[b_idx[pos], t_idx[pos], 2, y_cort[pos], x_canvas[pos]] = 1.0

    neg = pol < 0
    canvas[b_idx[neg], t_idx[neg], 0, y_cort[neg], x_canvas[neg]] = 1.0

    return canvas
