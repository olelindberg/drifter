#!/usr/bin/env python3
"""
Generate surface plots of Bezier control point bathymetry solutions.

Plots the actual Bezier control points (x, y, z) without interpolation.
Creates one plot per solver showing the fitted surface and error vs analytical.

Usage:
    cd scr && uv run surface_plots.py /tmp --output ../docs/figures
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


def load_surface_data(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all solver_surface_*.csv files."""
    data = {}
    for f in directory.glob("solver_surface_*.csv"):
        name = f.stem.replace("solver_surface_", "")
        data[name] = pd.read_csv(f)
    return data


def plot_surface_trisurf(
    df: pd.DataFrame,
    z_col: str,
    title: str,
    ax,
    cmap: str = "viridis",
    zlim: tuple | None = None,
):
    """Plot 3D surface using triangulation of scattered control points."""
    x = df["x"].values
    y = df["y"].values
    z = df[z_col].values

    # Create triangulation using matplotlib
    triang = mtri.Triangulation(x, y)

    # Plot surface using the triangulation
    surf = ax.plot_trisurf(
        triang,
        z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.9,
        linewidth=0,
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(title)

    if zlim is not None:
        ax.set_zlim(zlim)

    return surf


def plot_solver_surfaces(
    surfaces: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Generate 3D surface plot for each solver showing solution vs analytical."""
    solver_names = {
        "direct": "Direct (SparseLU)",
        "iterative_lu": "Iterative + LU",
        "iterative_mg": "Iterative + Multigrid",
        "analytical": "Analytical",
    }

    # Determine common z limits
    z_all = []
    for name, df in surfaces.items():
        if name == "analytical":
            z_all.extend(df["z_analytical"].values)
        else:
            z_all.extend(df["z_solution"].values)
            z_all.extend(df["z_analytical"].values)
    z_min, z_max = min(z_all), max(z_all)
    zlim = (z_min - 0.1 * abs(z_max - z_min), z_max + 0.1 * abs(z_max - z_min))

    for name, df in surfaces.items():
        if name == "analytical":
            continue  # Skip analytical-only file for main plots

        fig = plt.figure(figsize=(16, 6))

        # Plot 1: Solution surface
        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        plot_surface_trisurf(
            df, "z_solution", f"{solver_names.get(name, name)}: Solution", ax1, zlim=zlim
        )

        # Plot 2: Analytical surface
        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        plot_surface_trisurf(
            df, "z_analytical", "Analytical: cos(kx*x)*cos(ky*y)", ax2, zlim=zlim
        )

        # Plot 3: Error (solution - analytical)
        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        df_error = df.copy()
        df_error["z_error"] = df["z_solution"] - df["z_analytical"]
        error_max = df_error["z_error"].abs().max()
        plot_surface_trisurf(
            df_error,
            "z_error",
            f"Error (max: {error_max:.2e})",
            ax3,
            cmap="RdBu_r",
            zlim=(-error_max * 1.2, error_max * 1.2),
        )

        plt.tight_layout()

        if output_dir:
            output_dir.mkdir(exist_ok=True)
            filename = output_dir / f"surface_{name}.png"
            plt.savefig(filename, dpi=300, bbox_inches="tight")
            print(f"Saved: {filename}")
        else:
            plt.show()
        plt.close()


def plot_all_surfaces_comparison(
    surfaces: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Generate comparison plot showing all solver solutions side by side."""
    solver_order = ["direct", "iterative_lu", "iterative_mg"]
    solvers = [s for s in solver_order if s in surfaces]

    solver_names = {
        "direct": "Direct (SparseLU)",
        "iterative_lu": "Iterative + LU",
        "iterative_mg": "Iterative + MG",
    }

    if len(solvers) < 2:
        print("Not enough solvers for comparison plot")
        return

    # Determine common z limits from analytical
    z_all = []
    for name in solvers:
        df = surfaces[name]
        z_all.extend(df["z_analytical"].values)
    z_min, z_max = min(z_all), max(z_all)
    zlim = (z_min - 0.05 * abs(z_max - z_min), z_max + 0.05 * abs(z_max - z_min))

    # Create figure: one column per solver + analytical
    n_cols = len(solvers) + 1
    fig = plt.figure(figsize=(5 * n_cols, 5))

    # First: analytical (use any solver's data)
    ax = fig.add_subplot(1, n_cols, 1, projection="3d")
    plot_surface_trisurf(
        surfaces[solvers[0]], "z_analytical", "Analytical", ax, zlim=zlim
    )

    # Then: each solver's solution
    for idx, name in enumerate(solvers):
        ax = fig.add_subplot(1, n_cols, idx + 2, projection="3d")
        plot_surface_trisurf(
            surfaces[name],
            "z_solution",
            solver_names.get(name, name),
            ax,
            zlim=zlim,
        )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / "surface_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
    else:
        plt.show()
    plt.close()


def plot_error_comparison(
    surfaces: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Generate error comparison across all solvers."""
    solver_order = ["direct", "iterative_lu", "iterative_mg"]
    solvers = [s for s in solver_order if s in surfaces]

    solver_names = {
        "direct": "Direct",
        "iterative_lu": "Iter+LU",
        "iterative_mg": "Iter+MG",
    }

    if len(solvers) == 0:
        return

    # Compute max error for common color scale
    max_errors = {}
    for name in solvers:
        df = surfaces[name]
        error = (df["z_solution"] - df["z_analytical"]).abs().max()
        max_errors[name] = error

    global_max_error = max(max_errors.values())

    # Create figure
    fig = plt.figure(figsize=(5 * len(solvers), 5))

    for idx, name in enumerate(solvers):
        df = surfaces[name]
        df_error = df.copy()
        df_error["z_error"] = df["z_solution"] - df["z_analytical"]

        ax = fig.add_subplot(1, len(solvers), idx + 1, projection="3d")
        plot_surface_trisurf(
            df_error,
            "z_error",
            f"{solver_names[name]} Error (max: {max_errors[name]:.2e})",
            ax,
            cmap="RdBu_r",
            zlim=(-global_max_error * 1.2, global_max_error * 1.2),
        )

    plt.tight_layout()

    if output_dir:
        output_dir.mkdir(exist_ok=True)
        filename = output_dir / "surface_errors.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {filename}")
    else:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Surface plots of Bezier bathymetry")
    parser.add_argument("directory", help="Directory with CSV files")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    args = parser.parse_args()

    directory = Path(args.directory)
    output_dir = Path(args.output) if args.output else None

    # Load data
    surfaces = load_surface_data(directory)

    if not surfaces:
        print(f"No solver_surface_*.csv files found in {directory}")
        return

    print(f"Found surfaces: {list(surfaces.keys())}")

    # Generate plots
    plot_solver_surfaces(surfaces, output_dir)
    plot_all_surfaces_comparison(surfaces, output_dir)
    plot_error_comparison(surfaces, output_dir)

    if output_dir:
        print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
