"""Reproduce the 11 x 11 (121-channel) calculation used by Fig. 4."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from core_angular import Core


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate the paper's 121-channel angular grid.")
    parser.add_argument("--grid-size", type=int, default=11)
    parser.add_argument("--angle-mrad", type=float, default=12.0)
    parser.add_argument("--train-steps", type=int, default=500)
    parser.add_argument("--nx", type=int, default=800, help="Use 800 for paper reproduction.")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "targets")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "geometry_data_121_channels.mat",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_flags(args: argparse.Namespace) -> argparse.Namespace:
    px = py = 0.3e-6
    angle_degree = args.angle_mrad * 1e-3 * 180.0 / np.pi
    angle = ((np.arange(args.grid_size) - (args.grid_size - 1) / 2) * angle_degree).tolist()
    angle_x = angle * args.grid_size
    angle_y = [value for value in angle for _ in range(args.grid_size)]
    x = np.linspace(-(args.nx * px) / 2, (args.nx * px) / 2, args.nx + 1)
    y = np.linspace(-(args.nx * py) / 2, (args.nx * py) / 2, args.nx + 1)
    return argparse.Namespace(
        optim="Adam",
        train_step=args.train_steps,
        lr=40e-3,
        lr_decay_rate=0.8,
        data_dir=str(args.target_dir),
        function="AS",
        Nx=args.nx,
        Ny=args.nx,
        wavelength=[0.785e-6],
        ratio=0.3,
        x=x,
        y=y,
        x_ob=x,
        y_ob=y,
        refractive_index=[1.0, 1.45],
        distance=[200e-6, 600e-6],
        NA=None,
        eta=2,
        angle_x=angle_x,
        angle_y=angle_y,
        batch_size=args.grid_size,
    )


def main() -> None:
    args = parse_args()
    if args.grid_size < 2 or args.grid_size % 2 == 0:
        raise ValueError("--grid-size must be an odd integer of at least 3.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite explicitly.")
    if not args.target_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {args.target_dir}")

    flags = make_flags(args)
    print(
        f"Calculating {args.grid_size} x {args.grid_size} = {args.grid_size**2} channels "
        f"at delta_theta={args.angle_mrad:g} mrad",
        flush=True,
    )
    simulation = Core(flags)
    simulation.train()

    output_parts = []
    with torch.no_grad():
        for start in range(0, len(simulation.case_configs), args.grid_size):
            configs = simulation.case_configs[start : start + args.grid_size]
            output = simulation.model(configs)
            output = simulation.ratio_image(output)
            output_parts.append(np.abs(output.detach().cpu().numpy()) ** 2)
            print(f"Evaluated {start + len(configs)}/{len(simulation.case_configs)} channels", flush=True)

    output_field = np.concatenate(output_parts, axis=0)
    phase = simulation.list_from_gpu_to_cpu(simulation.MS_property)
    save_data = {
        "phase": phase,
        "output_field": output_field,
        "crosstalk": simulation.crosstalk.detach().cpu().numpy(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(args.output, save_data)
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
