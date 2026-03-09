#!/usr/bin/env python3
"""
Plot and compare CG Bezier solver verification metrics.

Usage:
    cd scr && uv run solver_verification.py /tmp
    cd scr && uv run solver_verification.py /tmp --output /tmp/solver_plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_final_metrics(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all solver_metrics_*.csv files."""
    data = {}
    for f in directory.glob("solver_metrics_*.csv"):
        name = f.stem.replace("solver_metrics_", "")
        data[name] = pd.read_csv(f, index_col="metric")
    return data


def load_iteration_history(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all solver_iterations_*.csv files (per-iteration metrics)."""
    data = {}
    for f in directory.glob("solver_iterations_*.csv"):
        name = f.stem.replace("solver_iterations_", "")
        data[name] = pd.read_csv(f)
    return data


def load_comparison(directory: Path) -> pd.DataFrame | None:
    """Load solver comparison CSV."""
    path = directory / "solver_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def compute_convergence_rate(df: pd.DataFrame) -> float:
    """Compute average convergence rate from residual history."""
    residuals = df["schur_residual_norm"].values
    if len(residuals) < 2:
        return 0.0
    # Average ratio of consecutive residuals
    ratios = residuals[1:] / residuals[:-1]
    ratios = ratios[ratios > 0]  # Filter invalid
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))


def plot_convergence(
    iterations: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Plot residual convergence history for iterative solvers."""
    if not iterations:
        print("No iteration history to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"iterative_lu": "#1f77b4", "iterative_mg": "#ff7f0e"}

    # Plot 1: Absolute residual norm
    ax1 = axes[0]
    for name, df in iterations.items():
        color = colors.get(name, None)
        ax1.semilogy(
            df["iteration"],
            df["schur_residual_norm"],
            marker="o",
            markersize=3,
            label=name,
            color=color,
        )
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Schur Residual L2 Norm")
    ax1.set_title("Convergence: Absolute Residual")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative residual
    ax2 = axes[1]
    for name, df in iterations.items():
        color = colors.get(name, None)
        ax2.semilogy(
            df["iteration"],
            df["relative_residual"],
            marker="o",
            markersize=3,
            label=name,
            color=color,
        )
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Relative Residual")
    ax2.set_title("Convergence: Relative Residual")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "solver_convergence.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'solver_convergence.png'}")
    else:
        plt.show()
    plt.close()


def plot_cg_parameters(
    iterations: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Plot alpha and pSp per iteration."""
    if not iterations:
        print("No iteration history to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"iterative_lu": "#1f77b4", "iterative_mg": "#ff7f0e"}

    # Plot alpha (step size)
    ax1 = axes[0]
    for name, df in iterations.items():
        color = colors.get(name, None)
        ax1.plot(df["iteration"], df["alpha"], marker=".", label=name, color=color)
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Alpha (step size)")
    ax1.set_title("CG Step Size per Iteration")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot pSp (curvature)
    ax2 = axes[1]
    for name, df in iterations.items():
        color = colors.get(name, None)
        ax2.semilogy(df["iteration"], df["pSp"], marker=".", label=name, color=color)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("p^T S p")
    ax2.set_title("Search Direction Curvature")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "solver_cg_parameters.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'solver_cg_parameters.png'}")
    else:
        plt.show()
    plt.close()


def plot_final_metrics_comparison(
    metrics: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Bar chart comparing final metrics across solvers."""
    if not metrics:
        print("No metrics to plot")
        return

    # Key metrics to plot
    metric_names = [
        "solution_l2_norm",
        "constraint_violation",
        "data_residual",
        "objective_value",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    solvers = list(metrics.keys())
    x = np.arange(len(solvers))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = []
        for s in solvers:
            try:
                values.append(metrics[s].loc[metric, "value"])
            except KeyError:
                values.append(0.0)

        bars = ax.bar(x, values, color=colors[: len(solvers)])
        ax.set_xticks(x)
        ax.set_xticklabels(solvers, rotation=15)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric)

        # Use log scale for constraint_violation
        if metric == "constraint_violation":
            ax.set_yscale("log")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if metric == "constraint_violation":
                label = f"{val:.2e}"
            else:
                label = f"{val:.4f}"
            ax.annotate(
                label,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "solver_final_metrics.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'solver_final_metrics.png'}")
    else:
        plt.show()
    plt.close()


def plot_timing_comparison(
    metrics: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Plot solve timing comparison."""
    if not metrics:
        return

    solvers = list(metrics.keys())
    times = []
    iterations = []

    for s in solvers:
        try:
            times.append(metrics[s].loc["total_solve_ms", "value"])
        except KeyError:
            times.append(0.0)
        try:
            iterations.append(int(metrics[s].loc["schur_cg_iterations", "value"]))
        except (KeyError, ValueError):
            iterations.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Timing
    ax1 = axes[0]
    bars = ax1.bar(solvers, times, color=colors[: len(solvers)])
    ax1.set_ylabel("Time (ms)")
    ax1.set_title("Total Solve Time")
    ax1.tick_params(axis="x", rotation=15)
    for bar, t in zip(bars, times):
        ax1.annotate(
            f"{t:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # Iterations
    ax2 = axes[1]
    bars = ax2.bar(solvers, iterations, color=colors[: len(solvers)])
    ax2.set_ylabel("Iterations")
    ax2.set_title("Schur CG Iterations")
    ax2.tick_params(axis="x", rotation=15)
    for bar, it in zip(bars, iterations):
        ax2.annotate(
            str(it),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / "solver_timing.png", dpi=300, bbox_inches="tight")
        print(f"Saved: {output_dir / 'solver_timing.png'}")
    else:
        plt.show()
    plt.close()


def print_summary_tables(
    metrics: dict[str, pd.DataFrame],
    iterations: dict[str, pd.DataFrame],
    comparison: pd.DataFrame | None,
) -> None:
    """Print formatted summary tables to console."""
    print("\n" + "=" * 70)
    print("                    FINAL METRICS COMPARISON")
    print("=" * 70)

    # Create a summary table
    if metrics:
        all_metrics = [
            "solution_l2_norm",
            "schur_cg_iterations",
            "data_residual",
            "regularization_energy",
            "constraint_violation",
            "objective_value",
            "qinv_apply_calls",
            "total_solve_ms",
        ]

        # Header
        solvers = list(metrics.keys())
        header = f"{'Metric':<25}" + "".join(f"{s:>20}" for s in solvers)
        print(header)
        print("-" * len(header))

        for metric in all_metrics:
            row = f"{metric:<25}"
            for s in solvers:
                try:
                    val = metrics[s].loc[metric, "value"]
                    if metric in ["schur_cg_iterations", "qinv_apply_calls"]:
                        row += f"{int(val):>20}"
                    elif metric == "constraint_violation":
                        row += f"{val:>20.2e}"
                    elif metric == "total_solve_ms":
                        row += f"{val:>20.2f}"
                    else:
                        row += f"{val:>20.6f}"
                except KeyError:
                    row += f"{'N/A':>20}"
            print(row)

    print("\n" + "=" * 70)
    print("                    CONVERGENCE SUMMARY")
    print("=" * 70)
    for name, df in iterations.items():
        print(f"\n{name}:")
        print(f"  Total iterations:   {len(df)}")
        if len(df) > 0:
            print(f"  Initial residual:   {df['schur_residual_norm'].iloc[0]:.6e}")
            print(f"  Final residual:     {df['schur_residual_norm'].iloc[-1]:.6e}")
            rate = compute_convergence_rate(df)
            print(f"  Avg conv. rate:     {rate:.4f}")

    if comparison is not None and len(comparison) > 0:
        print("\n" + "=" * 70)
        print("                    SOLUTION DIFFERENCES")
        print("=" * 70)
        print(f"{'Solver 1':<15} {'Solver 2':<15} {'L2 Diff':>15} {'Rel Diff':>15}")
        print("-" * 60)
        for _, row in comparison.iterrows():
            print(
                f"{row['solver1']:<15} {row['solver2']:<15} "
                f"{row['l2_diff']:>15.6e} {row['relative_diff']:>15.6e}"
            )


def main():
    parser = argparse.ArgumentParser(description="CG Bezier Solver Verification")
    parser.add_argument("directory", help="Directory with CSV files")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    args = parser.parse_args()

    directory = Path(args.directory)
    output_dir = Path(args.output) if args.output else None

    # Load data
    metrics = load_final_metrics(directory)
    iterations = load_iteration_history(directory)
    comparison = load_comparison(directory)

    if not metrics:
        print(f"No solver_metrics_*.csv files found in {directory}")
        return

    # Print summary
    print_summary_tables(metrics, iterations, comparison)

    # Generate plots
    if iterations:
        plot_convergence(iterations, output_dir)
        plot_cg_parameters(iterations, output_dir)
    plot_final_metrics_comparison(metrics, output_dir)
    plot_timing_comparison(metrics, output_dir)

    if output_dir:
        print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
