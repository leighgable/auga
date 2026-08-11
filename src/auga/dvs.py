import torch
import torch.nn as nn

import numpy as np


class SyntheticDVS:
    """
    Converts standard Atari frames (uint8, 84x84 or 210x160) into 
    synthetic Dynamic Vision Sensor (DVS) events via frame differencing.

    Output: stream of (x, y, t, p) events where p ∈ {+1 (ON), -1 (OFF)}.
    This mimics real event cameras like iniVation DAVIS or Prophesee sensors.
    """
    def __init__(
        self,
        frame_shape: tuple[int, int] = (84, 84),
        pos_threshold: float = 0.15,
        neg_threshold: float = 0.15,
        max_events_per_frame: int = 5000,
    ):
        self.frame_shape = frame_shape
        self.pos_threshold = pos_threshold
        self.neg_threshold = neg_threshold
        self.max_events = max_events_per_frame
        self.prev_frame = None

    def reset(self):
        self.prev_frame = None

    def frame_to_events(self, frame: np.ndarray) -> np.ndarray:
        """
        frame: np.ndarray of shape (H, W) or (H, W, C), uint8 or float
        Returns: events array of shape (N, 4) with columns [x, y, t, p]
        """
        # Normalize to [0, 1]
        if frame.dtype == np.uint8:
            frame = frame.astype(np.float32) / 255.0
        if frame.ndim == 3:
            frame = frame.mean(axis=2)  # grayscale

        # Resize to target shape
        if frame.shape != self.frame_shape:
            import cv2
            frame = cv2.resize(frame, self.frame_shape[::-1], interpolation=cv2.INTER_AREA)

        if self.prev_frame is None:
            self.prev_frame = frame
            return np.zeros((0, 4), dtype=np.float32)

        diff = frame - self.prev_frame
        self.prev_frame = frame.copy()

        # Generate events
        on_mask = diff > self.pos_threshold
        off_mask = diff < -self.neg_threshold

        on_yx = np.argwhere(on_mask)
        off_yx = np.argwhere(off_mask)

        events = []
        if len(on_yx) > 0:
            on_events = np.zeros((len(on_yx), 4), dtype=np.float32)
            on_events[:, 0] = on_yx[:, 1]  # x
            on_events[:, 1] = on_yx[:, 0]  # y
            on_events[:, 3] = 1.0  # ON polarity
            events.append(on_events)

        if len(off_yx) > 0:
            off_events = np.zeros((len(off_yx), 4), dtype=np.float32)
            off_events[:, 0] = off_yx[:, 1]
            off_events[:, 1] = off_yx[:, 0]
            off_events[:, 3] = -1.0  # OFF polarity
            events.append(off_events)

        if len(events) == 0:
            return np.zeros((0, 4), dtype=np.float32)

        all_events = np.concatenate(events, axis=0)

        # Subsample if too many
        if len(all_events) > self.max_events:
            idx = np.random.choice(len(all_events), self.max_events, replace=False)
            all_events = all_events[idx]

        return all_events


class EventAccumulator:
    """
    Accumulates DVS events over a temporal window into a 2-channel 
    event frame: [ON_events, OFF_events] of shape (2, H, W).

    This provides a dense tensor representation suitable for CNN encoding.
    """
    def __init__(self, frame_shape: tuple[int, int] = (84, 84), temporal_bins: int = 1):
        self.frame_shape = frame_shape
        self.temporal_bins = temporal_bins
        self.reset()

    def reset(self):
        self.buffer = []

    def accumulate(self, events: np.ndarray) -> torch.Tensor:
        """
        events: (N, 4) array [x, y, t, p]
        Returns: (2, H, W) tensor — accumulated ON/OFF event counts
        """
        if len(events) == 0:
            return torch.zeros(2, *self.frame_shape)

        H, W = self.frame_shape
        on_frame = np.zeros((H, W), dtype=np.float32)
        off_frame = np.zeros((H, W), dtype=np.float32)

        x = events[:, 0].astype(int)
        y = events[:, 1].astype(int)
        p = events[:, 3]

        # Clip to bounds
        x = np.clip(x, 0, W - 1)
        y = np.clip(y, 0, H - 1)

        for i in range(len(events)):
            if p[i] > 0:
                on_frame[y[i], x[i]] += 1.0
            else:
                off_frame[y[i], x[i]] += 1.0

        # Normalize by event count (optional, helps stability)
        event_count = len(events) + 1e-8
        on_frame = on_frame / event_count * 10.0  # scale factor
        off_frame = off_frame / event_count * 10.0

        frame = np.stack([on_frame, off_frame], axis=0)  # (2, H, W)
        return torch.from_numpy(frame).float()

    def update(self, events: np.ndarray) -> torch.Tensor:
        """Add events to buffer and return accumulated frame."""
        self.buffer.append(events)
        if len(self.buffer) > self.temporal_bins:
            self.buffer.pop(0)

        all_events = np.concatenate(self.buffer, axis=0) if self.buffer else np.zeros((0, 4))
        return self.accumulate(all_events)


class EventFrameEncoder(nn.Module):
    """
    CNN encoder that converts (2, H, W) event frames into 
    neural input I(t) ∈ R^{n_neurons}.

    Architecture: small CNN (Atari-style) -> FC -> n_neurons
    """
    def __init__(
        self,
        frame_shape: tuple[int, int] = (84, 84),
        out_dim: int = 256,
    ):
        super().__init__()
        H, W = frame_shape

        # Compute conv output size
        def conv_size(size, kernel, stride):
            return (size - kernel) // stride + 1

        h1, w1 = conv_size(H, 8, 4), conv_size(W, 8, 4)  # After conv1
        h2, w2 = conv_size(h1, 4, 2), conv_size(w1, 4, 2)  # After conv2
        h3, w3 = conv_size(h2, 3, 1), conv_size(w2, 3, 1)  # After conv3

        conv_out_dim = 64 * h3 * w3

        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_dim, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, event_frame: torch.Tensor) -> torch.Tensor:
        """
        event_frame: (batch, 2, H, W) or (2, H, W)
        Returns: (batch, out_dim) or (out_dim,)
        """
        if event_frame.dim() == 3:
            event_frame = event_frame.unsqueeze(0)
        x = self.conv(event_frame)
        x = self.fc(x)
        return x
