"""Reproduce the one-dimensional N=2, 4, and 8 angle sweep."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from core_angular import Core


ROOT = Path(__file__).resolve().parent
DEFAULT_ANGLES_MRAD = [0.1, 0.3, 0.57, 0.8, 1.084, 1.2, 1.399, 1.5, 1.8, 2.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate the paper's one-dimensional angular multiplexing sweep."
    )
    parser.add_argument("--n-values", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--angles-mrad", nargs="+", type=float, default=DEFAULT_ANGLES_MRAD)
    parser.add_argument("--train-steps", type=int, default=800)
    parser.add_argument("--nx", type=int, default=800, help="Use 800 for paper reproduction.")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "targets")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "simulation_sweep_results_N2_N4_N8.mat",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_flags(
    n: int,
    angle_mrad: float,
    train_steps: int,
    nx: int,
    target_dir: Path,
) -> argparse.Namespace:
    px = py = 0.3e-6
    angle_degree = angle_mrad * 1e-3 * 180.0 / np.pi
    full_list = [0.0] + [sign * k * angle_degree for k in range(1, n) for sign in (1, -1)]
    angle_x = full_list[:n]
    angle_y = [0.0] * n

    x = np.linspace(-(nx * px) / 2, (nx * px) / 2, nx + 1)
    y = np.linspace(-(nx * py) / 2, (nx * py) / 2, nx + 1)
    batch_size = n
    accumulation_steps = (n + batch_size - 1) // batch_size

    return argparse.Namespace(
        deta_angle=angle_degree,
        N=n,
        optim="Adam",
        train_step=train_steps,
        lr=30e-3,
        lr_decay_rate=0.8,
        stop_threshold=1e-6,
        data_dir=str(target_dir),
        function="AS",
        Nx=nx,
        Ny=nx,
        wavelength=[0.785e-6],
        wavelength_copy=[0.785e-6],
        ratio=0.5,
        x=x,
        y=y,
        x_ob=x,
        y_ob=y,
        refractive_index=[1.0, 1.45],
        distance=[200e-6, 450e-6],
        NA=None,
        eta=3,
        P_gap=6e-6,
        radius=0.4e-6,
        topology_vortex=[0],
        Angle_x=angle_x,
        Angle_y=angle_y,
        accumulation_steps=accumulation_steps,
        batch_size_per_gpu=batch_size,
        max_rotation_angle=0.0,
        max_distance_error=0.0,
        max_angle_error=0.0,
        rotation_error_mode="batch",
        distance_error_mode="batch",
        angle_error_mode="sample",
    )


def main() -> None:
    args = parse_args()
    if any(n < 2 for n in args.n_values):
        raise ValueError("Every N value must be at least 2.")
    if args.nx <= 0:
        raise ValueError("--nx must be positive.")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output}. Use --overwrite explicitly.")
    if not args.target_dir.is_dir():
        raise FileNotFoundError(f"Target directory not found: {args.target_dir}")

    results: list[dict[str, object]] = []
    for n in args.n_values:
        for angle_mrad in args.angles_mrad:
            print(f"\n=== N={n}, delta_theta={angle_mrad:g} mrad ===", flush=True)
            flags = make_flags(n, angle_mrad, args.train_steps, args.nx, args.target_dir)
            simulation = Core(flags)
            simulation.train()

            with torch.no_grad():
                output = simulation.model(simulation.case_configs)
                output = simulation.ratio_image(output)
                intensity = np.abs(output.detach().cpu().numpy()) ** 2

            results.append(
                {
                    "deta_angle": flags.deta_angle,
                    "deta_angle_mrad": angle_mrad,
                    "N": n,
                    "crosstalk_average": float(simulation.crosstalk_average.detach().cpu()),
                    "crosstalk_confusing": simulation.crosstalk.detach().cpu().numpy(),
                    "output_field1": intensity,
                }
            )
            del simulation, output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_data = {
        "deta_angle_list": np.asarray([r["deta_angle"] for r in results], dtype=np.float64),
        "deta_angle_mrad_list": np.asarray(
            [r["deta_angle_mrad"] for r in results], dtype=np.float64
        ),
        "N_list": np.asarray([r["N"] for r in results], dtype=np.int32),
        "crosstalk_average_list": np.asarray(
            [r["crosstalk_average"] for r in results], dtype=np.float32
        ),
        "crosstalk_confusing_cell": np.asarray(
            [r["crosstalk_confusing"] for r in results], dtype=object
        ),
        "output_field1_cell": np.asarray([r["output_field1"] for r in results], dtype=object),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(args.output, save_data)
    print(f"\nSaved {len(results)} cases to {args.output}")


if __name__ == "__main__":
    main()
