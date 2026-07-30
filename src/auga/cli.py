import time
from pathlib import Path
from typing import Optional

import torch
import torchvision
import numpy as np
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import gymnasium as gym
import ale_py

from fovea import isotropic, warp_to_isotropic, simulate_events

app = typer.Typer(
    name="foveated_vision_test",
    help="Atari Fovea Computer Vision Testing Pipeline",
    add_completion=False,
)

console = Console()

def save_video(path: Path, frames: list[torch.Tensor], fps: int) -> None:
    """
    save a list of (H, W, 3) uint8 CPU tensors as an MP4 via torchvision.

    requires ffmpeg in your PATH.
    """
    if not frames:
        console.print(f"[yellow]Warning:[/yellow] no frames to write to {path}")
        return

    # Stack to (T, H, W, C) uint8 — exactly what write_video expects
    video_tensor = torch.stack(frames, dim=0)
    torchvision.io.write_video(
        str(path),
        video_tensor,
        fps=fps,
        video_codec="libx264",
        options={"crf": "23", "preset": "fast"},
    )


@app.command()
def run(
    game: str = typer.Option(
        "ALE/Breakout-v5",
        "--game",
        "-g",
        help="ALE environment ID (e.g. ALE/Pong-v5, ALE/SpaceInvaders-v5)",
    ),
    total_frames: int = typer.Option(
        180,
        "--total-frames",
        "-n",
        min=2,
        help="Number of frames to capture",
    ),
    event_length: int = typer.Option(
        8,
        "--frame-length",
        "-f",
        min=2,
        help="Number of frames from which to aggregate events",
    ),
    event_threshold: float = typer.Option(
        0.15,
        "--event-threshold",
        "-et",
        help="Threshold for qualifying as an event"
    ),
    output: Path = typer.Option(
        Path("./fovea_output"),
        "--output",
        "-o",
        help="Output directory for videos and logs",
    ),
    video_fps: int = typer.Option(
        30,
        "--fps",
        help="FPS for output video files",
    ),
    aperture: float = typer.Option(
        0.25,
        "--aperture",
        "-a",
        min=0.0,
        max=1.0,
        help="Fovea sharp region radius",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        "-s",
        help="Random seed for environment",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        help="Torch device for the simulator (cpu, cuda, mps)",
    ),    
):
    gym.register_envs(ale_py)

    torch_device = torch.device(device)

        # Setup output directories
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    transformed_dir = output / "transformed"
    transformed_dir.mkdir(exist_ok=True)

    # Create environment
    env = gym.make(game, render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    # frame buffers
    original_frames = []
    event_frames = []

    # Timing buffers
    step_times: list[float] = []
    render_times: list[float] = []
    sim_times: list[float] = []

    total_start = time.perf_counter()
    actual_frames = 0

    console.print(f"[bold green]▶ Fovea Pipeline ... [/bold green]")
    console.print(f"  Game: [cyan]{game}[/cyan]")
    console.print(f"  Total Frames: [cyan]{total_frames}[/cyan]")
    console.print(f"  Resolution: [cyan]{w}x{h}[/cyan]")
    console.print(f"  Aperture: [cyan]{aperture}[/cyan]")
    console.print(f"  Output: [cyan]{output}[/cyan]")
    console.print(f"  Event Stack: [cyan]{event_length}[/cyan]")
    console.print(f"  Event Threshold: [cyan]{event_threshold}[/cyan]")

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Capturing frames...", total=total_frames)

            events = []

            for _ in range(total_frames):
                # Random action
                action = env.action_space.sample()

                # Environment step
                t0 = time.perf_counter()
                obs, reward, terminated, truncated, info = env.step(action)
                t1 = time.perf_counter()
                step_times.append(t1 - t0)

                # Render frame
                t0 = time.perf_counter()
                frame_np = env.render()
                t1 = time.perf_counter()
                render_times.append(t1 - t0)
                
                frame_t = torch.from_numpy(frame_np)
                original_frames.append(frame_t)

                # Events
                if len(events) % event_frames == 0:
                    t0 = time.perf_counter()
                    event = simulate_events(
                        events,
                        event_threshold,
                    )
                    t1 = time.perf_counter()
                    transform_times.append(t1 - t0)
                    events = []
                else:
                    events.append(frame_t)
                
                if event

                # Reset if episode ended
                if terminated or truncated:
                    obs, info = env.reset(seed=seed)

                progress.update(task, advance=1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
    finally:
        env.close()
        orig_writer.release()
        trans_writer.release()

        total_elapsed = time.perf_counter() - total_start
        actual_frames = len(step_times)

        def stats(times: list[float]) -> tuple:
            if not times:
                return 0.0, 0.0, 0.0
            arr = np.array(times)
            return float(np.mean(arr)), float(np.median(arr)), float(np.std(arr))

        step_mean, step_med, step_std = stats(step_times)
        render_mean, render_med, render_std = stats(render_times)
        trans_mean, trans_med, trans_std = stats(transform_times)
        write_mean, write_med, write_std = stats(write_times)

        total_frame_time = np.array(step_times) + np.array(render_times) + np.array(transform_times) + np.array(write_times)
        overall_mean = float(np.mean(total_frame_time))
        overall_fps = 1.0 / overall_mean if overall_mean > 0 else 0.0
        wall_fps = actual_frames / total_elapsed if total_elapsed > 0 else 0.0

        # Display results
        table = Table(title="[bold]Pipeline Timing Results[/bold]")
        table.add_column("Stage", style="cyan", no_wrap=True)
        table.add_column("Mean (ms)", justify="right")
        table.add_column("Median (ms)", justify="right")
        table.add_column("StdDev (ms)", justify="right")
        table.add_column("% of Frame", justify="right")

        def pct(t: float) -> str:
            return f"{(t / overall_mean * 100):.1f}%" if overall_mean > 0 else "N/A"

        table.add_row("Env Step", f"{step_mean*1000:.2f}", f"{step_med*1000:.2f}", f"{step_std*1000:.2f}", pct(step_mean))
        table.add_row("Render", f"{render_mean*1000:.2f}", f"{render_med*1000:.2f}", f"{render_std*1000:.2f}", pct(render_mean))
        table.add_row("Fovea Transform", f"{trans_mean*1000:.2f}", f"{trans_med*1000:.2f}", f"{trans_std*1000:.2f}", pct(trans_mean))
        table.add_row("Video Write", f"{write_mean*1000:.2f}", f"{write_med*1000:.2f}", f"{write_std*1000:.2f}", pct(write_mean))
        table.add_row("[bold]Total Frame[/bold]", f"[bold]{overall_mean*1000:.2f}[/bold]", "", "", "100%")

        console.print()
        console.print(table)

        summary = Table(title="[bold]Summary[/bold]")
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", style="green")

        summary.add_row("Frames Captured", str(actual_frames))
        summary.add_row("Wall Clock Time", f"{total_elapsed:.3f}s")
        summary.add_row("Pipeline FPS (mean)", f"{overall_fps:.2f}")
        summary.add_row("Wall FPS", f"{wall_fps:.2f}")
        summary.add_row("Original Video", str(orig_path))
        summary.add_row("Transformed Video", str(trans_path))

        console.print(summary)

        # Save timing log to file
        log_path = output / "timing.log"
        with open(log_path, "w") as f:
            f.write("Pipeline Timing Log\n")
            f.write("=========================\n")
            f.write(f"Game: {game}\n")
            f.write(f"Frames: {actual_frames}\n")
            f.write(f"Resolution: {w}x{h}\n")
            f.write(f"Fovea Radius: {fovea_radius}\n")
            f.write(f"Blur Kernel: {blur}\n")
            f.write(f"\n")
            f.write(f"Mean Frame Time: {overall_mean*1000:.3f} ms\n")
            f.write(f"Pipeline FPS: {overall_fps:.2f}\n")
            f.write(f"Wall FPS: {wall_fps:.2f}\n")
            f.write(f"\n")
            f.write(f"Per-Stage Breakdown (ms):\n")
            f.write(f"  Env Step:        {step_mean*1000:.3f} ± {step_std*1000:.3f}\n")
            f.write(f"  Render:          {render_mean*1000:.3f} ± {render_std*1000:.3f}\n")
            f.write(f"  Fovea Transform: {trans_mean*1000:.3f} ± {trans_std*1000:.3f}\n")
            f.write(f"  Video Write:     {write_mean*1000:.3f} ± {write_std*1000:.3f}\n")

        console.print(f"\n[dim]Timing log saved to: {log_path}[/dim]")


if __name__ == "__main__":
    app()
