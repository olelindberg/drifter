#!/usr/bin/env python3
"""
Plot synthetic bathymetry functions and adaptive meshes from exported CSV data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
from pathlib import Path


def load_bathymetry(csv_path):
    """Load bathymetry data from CSV and reshape to grid."""
    df = pd.read_csv(csv_path)
    n = int(np.sqrt(len(df)))
    x = df["x"].values.reshape(n, n)
    y = df["y"].values.reshape(n, n)
    z = df["depth"].values.reshape(n, n)
    return x, y, z


def load_mesh(csv_path):
    """Load mesh element bounds from CSV."""
    return pd.read_csv(csv_path)


def plot_bathymetry(x, y, z, title, output_path):
    """Create a surface plot of bathymetry data."""
    fig = plt.figure(figsize=(14, 5))

    # 3D surface plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot_surface(x, y, -z, cmap="terrain", linewidth=0, antialiased=True)
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Elevation (m)")
    ax1.set_title(f"{title} - 3D View")
    ax1.view_init(elev=35, azim=-60)

    # 2D contour plot
    ax2 = fig.add_subplot(122)
    levels = 20
    cf = ax2.contourf(x, y, z, levels=levels, cmap="terrain_r")
    ax2.contour(x, y, z, levels=levels, colors="k", linewidths=0.3, alpha=0.5)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title(f"{title} - Depth Contours")
    ax2.set_aspect("equal")
    cbar = plt.colorbar(cf, ax=ax2)
    cbar.set_label("Depth (m)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path.name}")


def plot_mesh(mesh_df, title, output_path, bathymetry_data=None):
    """Plot mesh elements as rectangles, optionally with bathymetry background."""
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot bathymetry contours as background if provided
    if bathymetry_data is not None:
        x, y, z = bathymetry_data
        ax.contourf(x, y, z, levels=20, cmap="terrain_r", alpha=0.5)

    # Create rectangles for each element
    patches = []
    colors = []
    max_level = mesh_df["level"].max()

    for _, row in mesh_df.iterrows():
        width = row["xmax"] - row["xmin"]
        height = row["ymax"] - row["ymin"]
        rect = Rectangle((row["xmin"], row["ymin"]), width, height)
        patches.append(rect)
        colors.append(row["level"])

    # Create collection with color based on refinement level
    pc = PatchCollection(patches, facecolor="none", edgecolor="black",
                         linewidth=0.5, alpha=0.8)
    ax.add_collection(pc)

    ax.set_xlim(mesh_df["xmin"].min(), mesh_df["xmax"].max())
    ax.set_ylim(mesh_df["ymin"].min(), mesh_df["ymax"].max())
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"{title}\n({len(mesh_df)} elements, max level {max_level})")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path.name}")


def plot_mesh_comparison(meshes, titles, output_path, bathymetry_data=None):
    """Plot multiple meshes in a grid with 2 columns for better visibility."""
    n = len(meshes)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 6 * nrows))
    axes = axes.flatten() if n > 1 else [axes]

    for ax, mesh_df, title in zip(axes[:n], meshes, titles):
        # Plot bathymetry contours as background if provided
        if bathymetry_data is not None:
            x, y, z = bathymetry_data
            ax.contourf(x, y, z, levels=15, cmap="terrain_r", alpha=0.4)

        # Create rectangles for each element
        patches = []
        for _, row in mesh_df.iterrows():
            width = row["xmax"] - row["xmin"]
            height = row["ymax"] - row["ymin"]
            rect = Rectangle((row["xmin"], row["ymin"]), width, height)
            patches.append(rect)

        pc = PatchCollection(patches, facecolor="none", edgecolor="black",
                             linewidth=0.5)
        ax.add_collection(pc)

        ax.set_xlim(mesh_df["xmin"].min(), mesh_df["xmax"].max())
        ax.set_ylim(mesh_df["ymin"].min(), mesh_df["ymax"].max())
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"{title}\n({len(mesh_df)} elem)")
        ax.set_aspect("equal")

    # Hide unused axes
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path.name}")


def plot_convergence(results_df, output_path):
    """Plot convergence: L2 error vs DOFs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    uniform = results_df[results_df["type"] == "uniform"]
    adaptive = results_df[results_df["type"] == "adaptive"]

    ax.loglog(uniform["dofs"].values, uniform["l2_error"].values, "o-",
              label="Uniform", markersize=8, linewidth=2)
    ax.loglog(adaptive["dofs"].values, adaptive["l2_error"].values, "s--",
              label="Adaptive", markersize=8, linewidth=2)

    ax.set_xlabel("Degrees of Freedom")
    ax.set_ylabel("L2 Error")
    ax.set_title("Convergence: Uniform vs Adaptive")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {output_path.name}")


def main():
    # Plot synthetic bathymetry
    input_dir = Path("/tmp/synthetic_bathymetry")
    output_dir = Path(__file__).parent.parent / "docs" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic bathymetry plots...")

    bathymetry_files = [
        ("canyon.csv", "Submarine Canyon", "synthetic_canyon.png"),
        ("seamount.csv", "Seamount", "synthetic_seamount.png"),
        ("ridge_system.csv", "Ridge System", "synthetic_ridge_system.png"),
        ("canyon_and_seamount.csv", "Canyon and Seamount", "synthetic_canyon_seamount.png"),
        ("shelf_break.csv", "Continental Shelf Break", "synthetic_shelf_break.png"),
    ]

    for csv_name, title, png_name in bathymetry_files:
        csv_path = input_dir / csv_name
        if csv_path.exists():
            x, y, z = load_bathymetry(csv_path)
            plot_bathymetry(x, y, z, title, output_dir / png_name)

    # Plot convergence study if available
    conv_dir = Path("/tmp/convergence_study")
    if conv_dir.exists():
        print("\nGenerating convergence study plots...")

        # Load bathymetry for background
        bathy_path = conv_dir / "bathymetry.csv"
        bathy_data = load_bathymetry(bathy_path) if bathy_path.exists() else None

        # Plot individual meshes
        mesh_files = sorted(conv_dir.glob("*_mesh.csv"))
        for mesh_file in mesh_files:
            mesh_df = load_mesh(mesh_file)
            name = mesh_file.stem.replace("_mesh", "")
            plot_mesh(mesh_df, name, output_dir / f"mesh_{name}.png", bathy_data)

        # Plot mesh comparison: uniform vs adaptive
        uniform_meshes = sorted(conv_dir.glob("uniform_*_mesh.csv"))
        adaptive_meshes = sorted(conv_dir.glob("adaptive_*_mesh.csv"))

        if uniform_meshes and adaptive_meshes:
            # Compare finest uniform with adaptive meshes
            meshes = [load_mesh(uniform_meshes[-1])] + [load_mesh(f) for f in adaptive_meshes]
            titles = [uniform_meshes[-1].stem.replace("_mesh", "").replace("uniform_", "Uniform ")] + \
                     [f.stem.replace("_mesh", "").replace("adaptive_", "Adaptive ") for f in adaptive_meshes]
            plot_mesh_comparison(meshes, titles, output_dir / "mesh_comparison.png", bathy_data)

        # Plot convergence results
        results_path = conv_dir / "convergence_results.csv"
        if results_path.exists():
            results_df = pd.read_csv(results_path)
            plot_convergence(results_df, output_dir / "convergence_uniform_vs_adaptive.png")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
