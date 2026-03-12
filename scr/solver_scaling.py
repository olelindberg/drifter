#!/usr/bin/env python3
"""
Plot solver scaling results: CPU time and iterations vs DOFs.

Usage:
    cd scr && uv run solver_scaling.py /tmp
    cd scr && uv run solver_scaling.py /tmp --output /tmp/plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Color mapping for all solver variants
SOLVER_COLORS = {
    "direct": "#1f77b4",
    "iterative_lu": "#ff7f0e",
    # Transfer/coarse strategy comparison
    "iterative_mg_l2_galerkin": "#2ca02c",
    "iterative_mg_l2_cached": "#d62728",
    "iterative_mg_bezier_galerkin": "#9467bd",
    "iterative_mg_bezier_cached": "#8c564b",
    # Coarsest level comparison variants
    "mg_coarse_4x4": "#2ca02c",
    "mg_coarse_8x8": "#d62728",
    "mg_coarse_16x16": "#9467bd",
}

# Marker mapping for all solver variants
SOLVER_MARKERS = {
    "direct": "s",  # square
    "iterative_lu": "^",  # triangle up
    # Transfer/coarse strategy comparison
    "iterative_mg_l2_galerkin": "o",  # circle
    "iterative_mg_l2_cached": "D",  # diamond
    "iterative_mg_bezier_galerkin": "v",  # triangle down
    "iterative_mg_bezier_cached": "p",  # pentagon
    # Coarsest level comparison variants
    "mg_coarse_4x4": "o",  # circle
    "mg_coarse_8x8": "D",  # diamond
    "mg_coarse_16x16": "v",  # triangle down
}

# Display names for solver variants
SOLVER_DISPLAY_NAMES = {
    "direct": "Direct",
    "iterative_lu": "Iterative+LU",
    # Transfer/coarse strategy comparison
    "iterative_mg_l2_galerkin": "MG (L2+Galerkin)",
    "iterative_mg_l2_cached": "MG (L2+Cached)",
    "iterative_mg_bezier_galerkin": "MG (Bezier+Galerkin)",
    "iterative_mg_bezier_cached": "MG (Bezier+Cached)",
    # Coarsest level comparison variants
    "mg_coarse_4x4": "MG (coarse=4×4)",
    "mg_coarse_8x8": "MG (coarse=8×8)",
    "mg_coarse_16x16": "MG (coarse=16×16)",
}

# Solvers for strategy comparison plot
STRATEGY_SOLVERS = [
    "direct",
    "iterative_lu",
    "iterative_mg_l2_galerkin",
    "iterative_mg_l2_cached",
    "iterative_mg_bezier_galerkin",
    "iterative_mg_bezier_cached",
]

# Solvers for coarsest level comparison plot
COARSEST_SOLVERS = [
    "direct",
    "iterative_lu",
    "mg_coarse_4x4",
    "mg_coarse_8x8",
    "mg_coarse_16x16",
]

# Preferred solver order for consistent plots (all solvers)
SOLVER_ORDER = STRATEGY_SOLVERS + [s for s in COARSEST_SOLVERS if s not in STRATEGY_SOLVERS]


def get_solver_color(name: str) -> str:
    """Get color for a solver, with fallback."""
    return SOLVER_COLORS.get(name, "#333333")


def get_solver_marker(name: str) -> str:
    """Get marker for a solver, with fallback."""
    return SOLVER_MARKERS.get(name, "o")


def get_solver_display_name(name: str) -> str:
    """Get display name for a solver."""
    return SOLVER_DISPLAY_NAMES.get(name, name)


def order_solvers(solver_names: list[str]) -> list[str]:
    """Order solvers according to preferred order."""
    ordered = []
    for s in SOLVER_ORDER:
        if s in solver_names:
            ordered.append(s)
    # Add any remaining solvers not in preferred order
    for s in solver_names:
        if s not in ordered:
            ordered.append(s)
    return ordered


def load_scaling_data(directory: Path) -> pd.DataFrame | None:
    """Load solver_scaling.csv file."""
    path = directory / "solver_scaling.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def plot_scaling_time_filtered(
    df: pd.DataFrame,
    output_dir: Path | None,
    solver_filter: list[str],
    filename: str,
    title: str,
) -> None:
    """Plot CPU time vs DOFs (log-log scale) for filtered solvers."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter to only solvers in the filter list that exist in data
    available_solvers = df["solver"].unique().tolist()
    solvers = [s for s in solver_filter if s in available_solvers]

    for solver in solvers:
        solver_df = df[df["solver"] == solver].sort_values("dofs")
        # Skip if no data
        if solver_df.empty:
            continue
        ax.loglog(
            solver_df["dofs"],
            solver_df["time_ms"],
            marker=get_solver_marker(solver),
            markersize=8,
            linewidth=2,
            label=get_solver_display_name(solver),
            color=get_solver_color(solver),
        )

    # Add reference lines for O(n) and O(n^2) scaling
    dofs = df["dofs"].unique()
    dofs_sorted = np.sort(dofs)
    if len(dofs_sorted) >= 2:
        # O(n) reference line (scaled to fit in plot)
        ref_time = df[df["dofs"] == dofs_sorted[0]]["time_ms"].mean()
        ref_dof = dofs_sorted[0]
        on_line = ref_time * (dofs_sorted / ref_dof)
        ax.loglog(dofs_sorted, on_line, "k--", alpha=0.3, linewidth=1.5, label="O(n)")

        # O(n^2) reference line
        on2_line = ref_time * (dofs_sorted / ref_dof) ** 2
        ax.loglog(
            dofs_sorted, on2_line, "k:", alpha=0.3, linewidth=1.5, label="O(n²)"
        )

    ax.set_xlabel("DOFs", fontsize=12)
    ax.set_ylabel("CPU Time (ms)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    # Add grid sizes as secondary x-axis labels
    ax2 = ax.twiny()
    grid_sizes = df.drop_duplicates("grid_size")[["grid_size", "dofs"]].sort_values(
        "dofs"
    )
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(grid_sizes["dofs"].tolist())
    ax2.set_xticklabels([f"{n}×{n}" for n in grid_sizes["grid_size"]])
    ax2.set_xlabel("Grid Size", fontsize=10)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / filename}")
    else:
        plt.show()
    plt.close()


def plot_scaling_time(df: pd.DataFrame, output_dir: Path | None) -> None:
    """Plot CPU time vs DOFs (log-log scale) for all solvers."""
    plot_scaling_time_filtered(
        df,
        output_dir,
        SOLVER_ORDER,
        "solver_scaling_time.png",
        "Solver Scaling: CPU Time vs Problem Size",
    )


def plot_scaling_strategy(df: pd.DataFrame, output_dir: Path | None) -> None:
    """Plot CPU time for transfer/coarse strategy comparison."""
    plot_scaling_time_filtered(
        df,
        output_dir,
        STRATEGY_SOLVERS,
        "solver_scaling_strategy.png",
        "Multigrid Strategy Comparison: Transfer Operators & Coarse Assembly",
    )


def plot_scaling_coarsest(df: pd.DataFrame, output_dir: Path | None) -> None:
    """Plot CPU time for coarsest level comparison."""
    plot_scaling_time_filtered(
        df,
        output_dir,
        COARSEST_SOLVERS,
        "solver_scaling_coarsest.png",
        "Coarsest Level Comparison: 4×4 vs 8×8 vs 16×16",
    )


def plot_scaling_iterations_filtered(
    df: pd.DataFrame,
    output_dir: Path | None,
    solver_filter: list[str],
    filename: str,
    title: str,
) -> None:
    """Plot Schur CG iterations vs DOFs for filtered solvers."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Filter out direct solver (has 0 iterations) and apply solver filter
    iterative_df = df[df["iterations"] > 0]
    available_solvers = iterative_df["solver"].unique().tolist()
    solvers = [s for s in solver_filter if s in available_solvers and s != "direct"]

    for solver in solvers:
        solver_df = iterative_df[iterative_df["solver"] == solver].sort_values("dofs")
        if solver_df.empty:
            continue
        ax.plot(
            solver_df["dofs"],
            solver_df["iterations"],
            marker=get_solver_marker(solver),
            markersize=8,
            linewidth=2,
            label=get_solver_display_name(solver),
            color=get_solver_color(solver),
        )

    ax.set_xlabel("DOFs", fontsize=12)
    ax.set_ylabel("Schur CG Iterations", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Use log scale for x-axis to match time plot
    ax.set_xscale("log")

    # Add grid sizes as secondary x-axis labels
    ax2 = ax.twiny()
    grid_sizes = df.drop_duplicates("grid_size")[["grid_size", "dofs"]].sort_values(
        "dofs"
    )
    ax2.set_xscale("log")
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(grid_sizes["dofs"].tolist())
    ax2.set_xticklabels([f"{n}×{n}" for n in grid_sizes["grid_size"]])
    ax2.set_xlabel("Grid Size", fontsize=10)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / filename}")
    else:
        plt.show()
    plt.close()


def plot_iterations_strategy(df: pd.DataFrame, output_dir: Path | None) -> None:
    """Plot iterations for transfer/coarse strategy comparison."""
    plot_scaling_iterations_filtered(
        df,
        output_dir,
        STRATEGY_SOLVERS,
        "solver_iterations_strategy.png",
        "Iteration Count: Strategy Comparison",
    )


def plot_iterations_coarsest(df: pd.DataFrame, output_dir: Path | None) -> None:
    """Plot iterations for coarsest level comparison."""
    plot_scaling_iterations_filtered(
        df,
        output_dir,
        COARSEST_SOLVERS,
        "solver_iterations_coarsest.png",
        "Iteration Count: Coarsest Level Comparison",
    )


def print_scaling_table(df: pd.DataFrame) -> None:
    """Print formatted scaling results table."""
    print("\n" + "=" * 100)
    print("                           SOLVER SCALING RESULTS")
    print("=" * 100)

    # Pivot table: rows=grid_size, columns=solver, values=time_ms
    pivot_time = df.pivot_table(
        index="grid_size", columns="solver", values="time_ms", aggfunc="first"
    )

    solvers = order_solvers(pivot_time.columns.tolist())
    pivot_time = pivot_time[solvers]

    print("\nCPU Time (ms):")
    col_width = 12
    header = f"{'Grid':>8}" + "".join(
        f"{get_solver_display_name(s):>{col_width}}" for s in solvers
    )
    print(header)
    print("-" * len(header))

    for grid_size in pivot_time.index:
        row = f"{grid_size:>6}×{grid_size:<1}"
        for solver in solvers:
            val = pivot_time.loc[grid_size, solver]
            row += f"{val:>{col_width}.1f}"
        print(row)

    # Pivot table for iterations
    pivot_iter = df.pivot_table(
        index="grid_size", columns="solver", values="iterations", aggfunc="first"
    )
    pivot_iter = pivot_iter[solvers]

    print("\nSchur CG Iterations:")
    print(header)
    print("-" * len(header))

    for grid_size in pivot_iter.index:
        row = f"{grid_size:>6}×{grid_size:<1}"
        for solver in solvers:
            val = pivot_iter.loc[grid_size, solver]
            if pd.isna(val):
                row += f"{'--':>{col_width}}"
            else:
                row += f"{int(val):>{col_width}d}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="CG Bezier Solver Scaling Plots")
    parser.add_argument("directory", help="Directory with solver_scaling.csv")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    args = parser.parse_args()

    directory = Path(args.directory)
    output_dir = Path(args.output) if args.output else None

    # Load data
    df = load_scaling_data(directory)

    if df is None or df.empty:
        print(f"No solver_scaling.csv found in {directory}")
        return

    # Print summary
    print_scaling_table(df)

    # Generate plots
    # 1. Strategy comparison (L2 vs Bezier, Galerkin vs Cached)
    plot_scaling_strategy(df, output_dir)
    plot_iterations_strategy(df, output_dir)
    # 2. Coarsest level comparison (4x4 vs 8x8 vs 16x16)
    plot_scaling_coarsest(df, output_dir)
    plot_iterations_coarsest(df, output_dir)

    if output_dir:
        print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
