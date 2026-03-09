#!/usr/bin/env python3
"""
Plot error distributions from adaptive CG Bezier refinement.

Reads CSV files written by AdaptiveCGLinearBezierSmoother when
error_output_dir is set, and generates histograms to visualize
the error distribution across iterations.

Usage:
    python plot_adaptive_errors.py /tmp/adaptive_errors
    python plot_adaptive_errors.py /tmp/adaptive_errors --metric weno_indicator
    python plot_adaptive_errors.py /tmp/adaptive_errors --iterations 0 5 10
"""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_iteration_data(directory: str) -> dict[int, pd.DataFrame]:
    """Load all error CSV files from a directory."""
    pattern = os.path.join(directory, "errors_iter_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No CSV files found matching {pattern}")
        sys.exit(1)

    data = {}
    for f in files:
        # Extract iteration number from filename
        basename = os.path.basename(f)
        iter_num = int(basename.replace("errors_iter_", "").replace(".csv", ""))
        data[iter_num] = pd.read_csv(f)

    return data


def print_statistics(df: pd.DataFrame, metric: str, iteration: int) -> None:
    """Print error statistics for a single iteration."""
    values = df[metric].values
    values = values[values > 0]  # Exclude zeros (land elements)

    if len(values) == 0:
        print(f"  Iteration {iteration}: No non-zero values")
        return

    print(f"  Iteration {iteration}:")
    print(f"    Count: {len(values)}")
    print(f"    Min:   {values.min():.6e}")
    print(f"    Max:   {values.max():.6e}")
    print(f"    Mean:  {values.mean():.6e}")
    print(f"    Median: {np.median(values):.6e}")
    print(f"    Std:   {values.std():.6e}")
    print(f"    P25:   {np.percentile(values, 25):.6e}")
    print(f"    P75:   {np.percentile(values, 75):.6e}")
    print(f"    P90:   {np.percentile(values, 90):.6e}")
    print(f"    P99:   {np.percentile(values, 99):.6e}")

    # Marked elements
    marked = df[df["marked"] == 1]
    print(f"    Marked: {len(marked)} elements")
    if len(marked) > 0:
        marked_values = marked[metric].values
        print(f"    Marked min: {marked_values.min():.6e}")
        print(f"    Marked max: {marked_values.max():.6e}")


def plot_histogram(
    data: dict[int, pd.DataFrame],
    metric: str,
    iterations: list[int] | None,
    output_file: str | None,
) -> None:
    """Plot histogram of error distribution."""
    if iterations is None:
        iterations = sorted(data.keys())

    n_plots = len(iterations)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    axes = axes[0]

    for idx, it in enumerate(iterations):
        if it not in data:
            print(f"Warning: iteration {it} not found")
            continue

        df = data[it]
        values = df[metric].values
        values = values[values > 0]  # Exclude zeros

        ax = axes[idx]

        # Use log scale for histogram
        if len(values) > 0:
            log_values = np.log10(values)
            ax.hist(log_values, bins=30, edgecolor="black", alpha=0.7)

            # Mark the marked elements
            marked_df = df[df["marked"] == 1]
            if len(marked_df) > 0:
                marked_values = marked_df[metric].values
                marked_values = marked_values[marked_values > 0]
                if len(marked_values) > 0:
                    min_marked = np.log10(marked_values.min())
                    ax.axvline(
                        min_marked,
                        color="red",
                        linestyle="--",
                        linewidth=2,
                        label=f"Marking cutoff",
                    )
                    ax.legend()

        ax.set_xlabel(f"log10({metric})")
        ax.set_ylabel("Count")
        ax.set_title(f"Iteration {it} ({len(df)} elements)")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_spatial(
    data: dict[int, pd.DataFrame],
    metric: str,
    iteration: int,
    output_file: str | None,
) -> None:
    """Plot spatial distribution of errors."""
    if iteration not in data:
        print(f"Iteration {iteration} not found")
        return

    df = data[iteration]

    fig, ax = plt.subplots(figsize=(8, 8))

    values = df[metric].values
    values = np.maximum(values, 1e-20)  # Avoid log(0)

    scatter = ax.scatter(
        df["center_x"],
        df["center_y"],
        c=np.log10(values),
        cmap="viridis",
        s=20,
        marker="s",
    )

    # Highlight marked elements
    marked = df[df["marked"] == 1]
    if len(marked) > 0:
        ax.scatter(
            marked["center_x"],
            marked["center_y"],
            facecolors="none",
            edgecolors="red",
            s=50,
            marker="o",
            linewidths=2,
            label="Marked",
        )
        ax.legend()

    plt.colorbar(scatter, ax=ax, label=f"log10({metric})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"Iteration {iteration}: {metric}")
    ax.set_aspect("equal")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot error distributions from adaptive refinement"
    )
    parser.add_argument("directory", help="Directory containing error CSV files")
    parser.add_argument(
        "--metric",
        default="normalized_error",
        choices=[
            "l2_error",
            "normalized_error",
            "mean_error",
            "std_error",
            "relative_error",
            "coarsening_error",
            "mean_difference",
            "volume_change",
        ],
        help="Error metric to plot",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        help="Specific iterations to plot (default: all)",
    )
    parser.add_argument("--output", "-o", help="Output file (default: show)")
    parser.add_argument(
        "--spatial",
        action="store_true",
        help="Plot spatial distribution instead of histogram",
    )
    parser.add_argument(
        "--spatial-iter",
        type=int,
        default=0,
        help="Iteration for spatial plot (default: 0)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print statistics only, no plot"
    )

    args = parser.parse_args()

    # Load data
    data = load_iteration_data(args.directory)
    print(f"Loaded {len(data)} iterations")

    # Print statistics
    print(f"\nStatistics for {args.metric}:")
    for it in sorted(data.keys()):
        print_statistics(data[it], args.metric, it)

    if args.stats:
        return

    # Plot
    if args.spatial:
        plot_spatial(data, args.metric, args.spatial_iter, args.output)
    else:
        plot_histogram(data, args.metric, args.iterations, args.output)


if __name__ == "__main__":
    main()
