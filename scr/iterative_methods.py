#!/usr/bin/env python3
"""
Plot and compare standalone iterative method verification metrics.

Usage:
    cd scr && uv run iterative_methods.py /tmp
    cd scr && uv run iterative_methods.py /tmp --output /tmp/method_plots
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Method display names and colors
METHOD_STYLES = {
    "jacobi": {"label": "Jacobi", "color": "#1f77b4", "marker": "o"},
    "multiplicative_schwarz": {
        "label": "Multiplicative Schwarz",
        "color": "#ff7f0e",
        "marker": "s",
    },
    "additive_schwarz": {"label": "Additive Schwarz", "color": "#2ca02c", "marker": "^"},
    "colored_schwarz": {"label": "Colored Schwarz", "color": "#d62728", "marker": "D"},
}


def load_iteration_history(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all method_iterations_*.csv files."""
    data = {}
    for f in directory.glob("method_iterations_*.csv"):
        name = f.stem.replace("method_iterations_", "")
        df = pd.read_csv(f)
        if len(df) > 0:
            data[name] = df
    return data


def load_comparison(directory: Path) -> pd.DataFrame | None:
    """Load method comparison CSV."""
    path = directory / "method_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def compute_convergence_rate(residuals: np.ndarray) -> float:
    """Compute average convergence rate from residual history."""
    if len(residuals) < 2:
        return 0.0
    # Average ratio of consecutive residuals
    ratios = residuals[1:] / residuals[:-1]
    ratios = ratios[(ratios > 0) & (ratios < 10)]  # Filter invalid/diverging
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))


def plot_convergence(
    iterations: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Plot residual convergence history for all methods."""
    if not iterations:
        print("No iteration history to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in sorted(iterations.items()):
        style = METHOD_STYLES.get(name, {"label": name, "color": "gray", "marker": "."})
        ax.semilogy(
            df["iteration"],
            df["residual_norm"],
            marker=style["marker"],
            markersize=4,
            markevery=max(1, len(df) // 20),
            label=style["label"],
            color=style["color"],
            linewidth=1.5,
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Residual L2 Norm", fontsize=12)
    ax.set_title("Convergence of Standalone Iterative Methods", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        output_dir.mkdir(exist_ok=True)
        filepath = output_dir / "method_convergence.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()
    plt.close()


def plot_convergence_detail(
    iterations: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Plot detailed convergence for fast-converging methods (first 20 iterations)."""
    if not iterations:
        return

    # Filter to methods that converge quickly
    fast_methods = {
        k: v for k, v in iterations.items() if len(v) <= 100 and k != "additive_schwarz"
    }

    if not fast_methods:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in sorted(fast_methods.items()):
        style = METHOD_STYLES.get(name, {"label": name, "color": "gray", "marker": "."})
        # Show only first 20 iterations for detail
        df_head = df.head(20)
        ax.semilogy(
            df_head["iteration"],
            df_head["residual_norm"],
            marker=style["marker"],
            markersize=6,
            label=style["label"],
            color=style["color"],
            linewidth=2,
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Residual L2 Norm", fontsize=12)
    ax.set_title("Early Convergence Detail (First 20 Iterations)", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        filepath = output_dir / "method_convergence_detail.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()
    plt.close()


def plot_comparison_bars(
    comparison: pd.DataFrame | None, output_dir: Path | None
) -> None:
    """Plot bar charts comparing methods."""
    if comparison is None or len(comparison) == 0:
        print("No comparison data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    methods = comparison["method"].tolist()
    colors = [METHOD_STYLES.get(m, {}).get("color", "gray") for m in methods]
    labels = [METHOD_STYLES.get(m, {}).get("label", m) for m in methods]

    # Plot 1: Iterations
    ax = axes[0, 0]
    bars = ax.bar(labels, comparison["iterations"], color=colors)
    ax.set_ylabel("Iterations")
    ax.set_title("Iterations to Converge")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, comparison["iterations"]):
        ax.annotate(
            str(val),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Plot 2: Relative Residual
    ax = axes[0, 1]
    bars = ax.bar(labels, comparison["relative_residual"], color=colors)
    ax.set_ylabel("Relative Residual")
    ax.set_title("Final Relative Residual")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, comparison["relative_residual"]):
        ax.annotate(
            f"{val:.1e}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 3: Solution Error
    ax = axes[1, 0]
    bars = ax.bar(labels, comparison["solution_error"], color=colors)
    ax.set_ylabel("Solution Error (rel.)")
    ax.set_title("Solution Error vs Reference")
    ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, comparison["solution_error"]):
        ax.annotate(
            f"{val:.1e}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 4: Timing
    ax = axes[1, 1]
    bars = ax.bar(labels, comparison["total_ms"], color=colors)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Total Solve Time")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, comparison["total_ms"]):
        ax.annotate(
            f"{val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if output_dir:
        filepath = output_dir / "method_comparison.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()
    plt.close()


def plot_convergence_rates(
    iterations: dict[str, pd.DataFrame], output_dir: Path | None
) -> None:
    """Plot convergence rate per iteration."""
    if not iterations:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for name, df in sorted(iterations.items()):
        if len(df) < 2:
            continue
        style = METHOD_STYLES.get(name, {"label": name, "color": "gray", "marker": "."})

        residuals = df["residual_norm"].values
        # Compute per-iteration convergence rate
        rates = residuals[1:] / residuals[:-1]
        # Filter out diverging steps
        rates = np.clip(rates, 0, 2)

        ax.plot(
            df["iteration"].values[1:],
            rates,
            marker=style["marker"],
            markersize=3,
            markevery=max(1, len(rates) // 20),
            label=style["label"],
            color=style["color"],
            linewidth=1,
            alpha=0.8,
        )

    ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1, alpha=0.5, label="No progress")
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Convergence Rate (r_{k+1}/r_k)", fontsize=12)
    ax.set_title("Per-Iteration Convergence Rate", fontsize=14)
    ax.set_ylim(0, 1.5)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_dir:
        filepath = output_dir / "method_convergence_rate.png"
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        print(f"Saved: {filepath}")
    else:
        plt.show()
    plt.close()


def print_summary(
    iterations: dict[str, pd.DataFrame], comparison: pd.DataFrame | None
) -> None:
    """Print formatted summary to console."""
    print("\n" + "=" * 80)
    print("              STANDALONE ITERATIVE METHODS SUMMARY")
    print("=" * 80)

    if comparison is not None:
        print(
            f"\n{'Method':<25} {'Iters':>8} {'Rel.Resid':>12} "
            f"{'Sol.Error':>12} {'Time(ms)':>10} {'Conv?':>6}"
        )
        print("-" * 80)
        for _, row in comparison.iterrows():
            label = METHOD_STYLES.get(row["method"], {}).get("label", row["method"])
            conv = "Yes" if str(row["converged"]).lower() == "true" else "No"
            print(
                f"{label:<25} {row['iterations']:>8} {row['relative_residual']:>12.2e} "
                f"{row['solution_error']:>12.2e} {row['total_ms']:>10.2f} {conv:>6}"
            )

    print("\n" + "=" * 80)
    print("              CONVERGENCE ANALYSIS")
    print("=" * 80)
    for name, df in sorted(iterations.items()):
        label = METHOD_STYLES.get(name, {}).get("label", name)
        residuals = df["residual_norm"].values
        rate = compute_convergence_rate(residuals)
        reduction = residuals[0] / residuals[-1] if residuals[-1] > 0 else float("inf")

        print(f"\n{label}:")
        print(f"  Iterations:         {len(df)}")
        print(f"  Initial residual:   {residuals[0]:.6e}")
        print(f"  Final residual:     {residuals[-1]:.6e}")
        print(f"  Total reduction:    {reduction:.2e}x")
        print(f"  Avg conv. rate:     {rate:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Iterative Methods Visualization"
    )
    parser.add_argument("directory", help="Directory with CSV files")
    parser.add_argument("--output", "-o", help="Output directory for plots")
    args = parser.parse_args()

    directory = Path(args.directory)
    output_dir = Path(args.output) if args.output else None

    # Load data
    iterations = load_iteration_history(directory)
    comparison = load_comparison(directory)

    if not iterations:
        print(f"No method_iterations_*.csv files found in {directory}")
        return

    # Print summary
    print_summary(iterations, comparison)

    # Generate plots
    plot_convergence(iterations, output_dir)
    plot_convergence_detail(iterations, output_dir)
    plot_convergence_rates(iterations, output_dir)
    if comparison is not None:
        plot_comparison_bars(comparison, output_dir)

    if output_dir:
        print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
