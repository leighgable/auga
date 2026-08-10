#!/usr/bin/env python3
"""
Atari Fovea Vision Pipeline CLI

Captures frames from an ALE environment, simulates an event camera by
diffing log-luminosity across N frames, and saves original, cartesian,
and isotropic event videos.

All image/video I/O uses PyTorch / torchvision / torchcodec. No OpenCV yet.
"""

import time
from pathlib import Path
from typing import Optional
import subprocess

import torch
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
)
import gymnasium as gym
import ale_py

from fovea import isotropic, simulate_events, render_cartesian_events, render_isotropic_events

app = typer.Typer(
    name="foveated_vision_test",
    help="Atari Fovea Computer Vision Testing Pipeline",
    add_completion=False,
)
console = Console()

def save_video_piped(path: Path, frames: list[torch.Tensor], fps: int) -> None:
    if not frames:
        return
    video = torch.stack(frames, dim=0).contiguous().byte()
    T, H, W, C = video.shape

    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-pixel_format", "rgb24",
            "-video_size", f"{W}x{H}",
            "-framerate", str(fps),
            "-i", "-",                      # read from stdin
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "23",
            "-preset", "fast",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdin.write(video.numpy().tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.read().decode())


def rendered_to_frames(rendered: torch.Tensor) -> list[torch.Tensor]:
    """
    Convert a (B, T, 3, H, W) float [0,1] tensor to a flat list of
    (H, W, 3) uint8 CPU tensors for video encoding.
    """
    if rendered.numel() == 0:
        return []
    # (B, T, 3, H, W) -> (B*T, H, W, 3) uint8
    flat = (
        rendered
        .permute(0, 1, 3, 4, 2)   # (B, T, H, W, 3)
        .reshape(-1, rendered.shape[3], rendered.shape[4], 3)
        .clamp(0, 1)
        .mul(255)
        .byte()
        .cpu()
    )
    return [flat[i] for i in range(flat.shape[0])]


@app.command()
def run(
    game: str = typer.Option(
        "ALE/Breakout-v5",
        "--game",
        "-g",
        help="ALE environment ID",
    ),
    total_frames: int = typer.Option(
        180,
        "--total-frames",
        "-n",
        min=2,
        help="Number of raw frames to capture",
    ),
    event_length: int = typer.Option(
        8,
        "--frame-length",
        "-f",
        min=2,
        help="Frames per event chunk",
    ),
    event_threshold: float = typer.Option(
        0.15,
        "--event-threshold",
        "-et",
        help="Contrast threshold for event detection",
    ),
    output: Path = typer.Option(
        Path("./fovea_output"),
        "--output",
        "-o",
        help="Output directory",
    ),
    video_fps: int = typer.Option(
        30,
        "--fps",
        help="FPS for output videos",
    ),
    aperture: float = typer.Option(
        0.25,
        "--aperture",
        "-a",
        min=0.0,
        max=1.0,
        help="Foveal compression factor",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device",
    ),
):
    gym.register_envs(ale_py)
    torch_device = torch.device(device)

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    transformed_dir = output / "transformed"
    transformed_dir.mkdir(exist_ok=True)

    env = gym.make(game, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    first_frame = env.render()
    H, W = first_frame.shape[:2]

    # Frame buffers
    original_frames: list[torch.Tensor] = []
    cartesian_event_frames: list[torch.Tensor] = []
    isotropic_event_frames: list[torch.Tensor] = []
    event_buffer: list[torch.Tensor] = []

    # Prepend the first frame
    original_frames.append(torch.from_numpy(first_frame))

    # Timing
    step_times: list[float] = []
    render_times: list[float] = []
    transform_times: list[float] = []

    total_start = time.perf_counter()
    chunks_processed = 0

    safe_name = game.replace("/", "_").replace("\\", "_")
    original_path = output / f"{safe_name}_original.mp4"
    cartesian_path = transformed_dir / f"{safe_name}_cartesian.mp4"
    isotropic_path = transformed_dir / f"{safe_name}_isotropic.mp4"

    console.print(f"[bold green]▶ Fovea Pipeline[/bold green]")
    console.print(f"  Game:          [cyan]{game}[/cyan]")
    console.print(f"  Resolution:    [cyan]{W}x{H}[/cyan]")
    console.print(f"  Total Frames:  [cyan]{total_frames}[/cyan]")
    console.print(f"  Event Length:  [cyan]{event_length}[/cyan]")
    console.print(f"  Aperture:      [cyan]{aperture}[/cyan]")
    console.print(f"  Threshold:     [cyan]{event_threshold}[/cyan]")
    console.print(f"  Output:        [cyan]{output}[/cyan]")
    console.print()

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Capturing frames...", total=total_frames)

            for i in range(1, total_frames):
                # Environment step
                t0 = time.perf_counter()
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                t1 = time.perf_counter()
                step_times.append(t1 - t0)

                # Render
                t0 = time.perf_counter()
                frame_np = env.render()
                t1 = time.perf_counter()
                render_times.append(t1 - t0)

                frame_t = torch.from_numpy(frame_np)
                original_frames.append(frame_t)
                event_buffer.append(frame_t)

                # Process chunk when full
                if len(event_buffer) == event_length:
                    t0 = time.perf_counter()

                    chunk = (
                        torch.stack(event_buffer, dim=0)   # (T, H, W, 3)
                        .permute(0, 3, 1, 2)              # (T, 3, H, W)
                        .unsqueeze(0)                     # (1, T, 3, H, W)
                        .float()
                        .div(255.0)
                        .to(torch_device)
                    )

                    _, sparse = simulate_events(chunk, event_threshold)

                    if sparse.shape[0] > 0:
                        iso_coords, hemi_idx = isotropic(
                            sparse_coords=sparse,
                            spatial_dims=(H, W),
                            a=aperture,
                        )
                        cartesian_rendered = render_cartesian_events(
                            sparse_coords=sparse,
                            frame_shape=(1, event_length - 1, H, W),
                        )
                        isotropic_rendered = render_isotropic_events(
                            sparse_coords=sparse,
                            isotropic_coords=iso_coords,
                            hemisphere_idx=hemi_idx,
                            frame_shape=(1, event_length, H, W),
                            isotropic_dims=(64, 64),
                            a=0.05,
                        )

                        cartesian_event_frames.extend(rendered_to_frames(cartesian_rendered))
                        isotropic_event_frames.extend(rendered_to_frames(isotropic_rendered))
                        chunks_processed += 1

                    t1 = time.perf_counter()
                    transform_times.append(t1 - t0)
                    event_buffer = []

                # episode reset
                if terminated or truncated:
                    obs, info = env.reset(seed=seed)
                    event_buffer = []  # discard partial chunk

                progress.update(task, advance=1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    finally:
        env.close()

        # Encode videos
        with console.status("[bold green]Encoding videos..."):
            save_video_piped(original_path, original_frames, video_fps)
            save_video_piped(cartesian_path, cartesian_event_frames, video_fps)
            save_video_piped(isotropic_path, isotropic_event_frames, video_fps)

        total_elapsed = time.perf_counter() - total_start
        raw_frames = len(original_frames)

        def stats(times: list[float]) -> tuple:
            if not times:
                return 0.0, 0.0, 0.0
            arr = np.array(times)
            return float(arr.mean()), float(np.median(arr)), float(arr.std())

        step_mean, step_med, step_std = stats(step_times)
        render_mean, render_med, render_std = stats(render_times)
        trans_mean, trans_med, trans_std = stats(transform_times)

        # transform time over raw frames tb fair
        trans_per_raw = (trans_mean * len(transform_times)) / raw_frames if raw_frames > 0 else 0.0
        overall_mean = step_mean + render_mean + trans_per_raw
        overall_fps = 1.0 / overall_mean if overall_mean > 0 else 0.0
        wall_fps = raw_frames / total_elapsed if total_elapsed > 0 else 0.0

        table = Table(title="[bold]Pipeline Timing Results[/bold]")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Mean (ms)", justify="right")
        table.add_column("Median (ms)", justify="right")
        table.add_column("StdDev (ms)", justify="right")
        table.add_column("% of Frame", justify="right")

        def pct(t: float) -> str:
            return f"{(t / overall_mean * 100):.1f}%" if overall_mean > 0 else "N/A"

        table.add_row("Env Step",   f"{step_mean*1000:.2f}", f"{step_med*1000:.2f}", f"{step_std*1000:.2f}", pct(step_mean))
        table.add_row("Render",     f"{render_mean*1000:.2f}", f"{render_med*1000:.2f}", f"{render_std*1000:.2f}", pct(render_mean))
        table.add_row("Event Sim",  f"{trans_mean*1000:.2f}", f"{trans_med*1000:.2f}", f"{trans_std*1000:.2f}", pct(trans_per_raw))
        table.add_row("[bold]Total[/bold]", f"[bold]{overall_mean*1000:.2f}[/bold]", "", "", "100%")

        console.print()
        console.print(table)

        summary = Table(title="[bold]Summary[/bold]")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Raw Frames",       str(raw_frames))
        summary.add_row("Chunks Processed", str(chunks_processed))
        summary.add_row("Wall Clock Time",  f"{total_elapsed:.3f}s")
        summary.add_row("Pipeline FPS",     f"{overall_fps:.2f}")
        summary.add_row("Wall FPS",         f"{wall_fps:.2f}")
        summary.add_row("Original Video",   str(original_path))
        summary.add_row("Cartesian Events", str(cartesian_path))
        summary.add_row("Isotropic Events", str(isotropic_path))

        console.print(summary)

        log_path = output / "timing.log"
        with open(log_path, "w") as f:
            f.write("Fovea Pipeline Timing Log\n")
            f.write("=========================\n")
            f.write(f"Game:     {game}\n")
            f.write(f"Frames:   {raw_frames}\n")
            f.write(f"Chunks:   {chunks_processed}\n")
            f.write(f"\n")
            f.write(f"Pipeline FPS: {overall_fps:.2f}\n")
            f.write(f"Wall FPS:     {wall_fps:.2f}\n")
            f.write(f"\n")
            f.write(f"Per-Stage Breakdown (ms):\n")
            f.write(f"  Env Step:    {step_mean*1000:.3f} ± {step_std*1000:.3f}\n")
            f.write(f"  Render:      {render_mean*1000:.3f} ± {render_std*1000:.3f}\n")
            f.write(f"  Event Sim:   {trans_mean*1000:.3f} ± {trans_std*1000:.3f}\n")

        console.print(f"\n[dim]Timing log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    app()
