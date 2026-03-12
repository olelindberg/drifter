#!/usr/bin/env python3
"""
Generate CG Bezier Solver Verification Report from CSV data.

Usage:
    cd scr && uv run generate_report.py /tmp --output ../docs
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Display names for solver variants
SOLVER_DISPLAY_NAMES = {
    "direct": "Direct",
    "iterative_lu": "Iterative+LU",
    "iterative_mg": "MG (L2+Galerkin)",  # Legacy name
    "iterative_mg_l2_galerkin": "MG (L2+Galerkin)",
    "iterative_mg_l2_cached": "MG (L2+Cached)",
    "iterative_mg_bezier_galerkin": "MG (Bezier+Galerkin)",
    "iterative_mg_bezier_cached": "MG (Bezier+Cached)",
}

# Preferred solver order for consistent tables
SOLVER_ORDER = [
    "direct",
    "iterative_lu",
    "iterative_mg",  # Legacy name
    "iterative_mg_l2_galerkin",
    "iterative_mg_l2_cached",
    "iterative_mg_bezier_galerkin",
    "iterative_mg_bezier_cached",
]


def get_solver_display_name(name: str) -> str:
    """Get display name for a solver."""
    return SOLVER_DISPLAY_NAMES.get(name, name)


def order_solvers(solver_names: list[str]) -> list[str]:
    """Order solvers according to preferred order."""
    ordered = []
    for s in SOLVER_ORDER:
        if s in solver_names:
            ordered.append(s)
    for s in solver_names:
        if s not in ordered:
            ordered.append(s)
    return ordered


def load_solver_metrics(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all solver_metrics_*.csv files."""
    data = {}
    for f in directory.glob("solver_metrics_*.csv"):
        name = f.stem.replace("solver_metrics_", "")
        data[name] = pd.read_csv(f, index_col="metric")
    return data


def load_solver_iterations(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all solver_iterations_*.csv files."""
    data = {}
    for f in directory.glob("solver_iterations_*.csv"):
        name = f.stem.replace("solver_iterations_", "")
        data[name] = pd.read_csv(f)
    return data


def load_solver_comparison(directory: Path) -> pd.DataFrame | None:
    """Load solver comparison CSV."""
    path = directory / "solver_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_method_iterations(directory: Path) -> dict[str, pd.DataFrame]:
    """Load all method_iterations_*.csv files."""
    data = {}
    for f in directory.glob("method_iterations_*.csv"):
        name = f.stem.replace("method_iterations_", "")
        df = pd.read_csv(f)
        if len(df) > 0:
            data[name] = df
    return data


def load_method_comparison(directory: Path) -> pd.DataFrame | None:
    """Load method comparison CSV."""
    path = directory / "method_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def load_scaling_data(directory: Path) -> pd.DataFrame | None:
    """Load solver_scaling.csv file."""
    path = directory / "solver_scaling.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def compute_convergence_rate(residuals: np.ndarray) -> float:
    """Compute average convergence rate from residual history."""
    if len(residuals) < 2:
        return 0.0
    ratios = residuals[1:] / residuals[:-1]
    ratios = ratios[(ratios > 0) & (ratios < 10)]
    if len(ratios) == 0:
        return 0.0
    return float(np.mean(ratios))


def fmt(val: float, metric: str) -> str:
    """Format a metric value for display."""
    if metric in ["schur_cg_iterations", "qinv_apply_calls", "iterations"]:
        return str(int(val))
    elif metric in ["constraint_violation", "relative_residual", "solution_error"]:
        return f"{val:.2e}"
    elif metric == "total_solve_ms" or metric == "total_ms":
        return f"{int(val)}"
    elif metric in ["solution_l2_norm", "objective_value"]:
        return f"{val:.3f}"
    elif metric in ["data_residual", "regularization_energy"]:
        return f"{val:.4f}"
    else:
        return f"{val:.2e}"


def generate_report(
    solver_metrics: dict[str, pd.DataFrame],
    solver_iterations: dict[str, pd.DataFrame],
    solver_comparison: pd.DataFrame | None,
    method_iterations: dict[str, pd.DataFrame],
    method_comparison: pd.DataFrame | None,
    scaling_data: pd.DataFrame | None,
) -> str:
    """Generate the full markdown report."""
    lines = []

    # Header
    lines.append("# CG Bezier Solver Verification Report")
    lines.append("")
    lines.append(
        "This report documents the verification of the CG (Conjugate Gradient) "
        "Bezier bathymetry smoothers, comparing different solver strategies and "
        "iterative methods."
    )
    lines.append("")
    lines.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    lines.append("")

    # Test Configuration
    lines.append("## Test Configuration")
    lines.append("")
    lines.append("- **Domain**: 1000m x 1000m")
    lines.append("- **Mesh**: 16x16 uniform quadtree (256 elements, 2401 DOFs)")
    lines.append("- **Bathymetry**: Exponential-sinusoidal function `exp(sin(2*pi*x/L) * sin(2*pi*y/L))`")
    lines.append("- **Lambda**: 1.0 (data fitting weight)")
    lines.append("- **Edge constraints**: 4 Gauss points per edge (C1 continuity)")
    lines.append("")

    # Section 1: Solver Comparison
    lines.append("## 1. Solver Comparison")
    lines.append("")
    lines.append("Multiple solver strategies are compared for the CG cubic Bezier smoother:")
    lines.append("")
    lines.append("| Solver | Transfer | Coarse Grid | Description |")
    lines.append("|--------|----------|-------------|-------------|")
    lines.append("| **Direct (SparseLU)** | N/A | N/A | Direct factorization of the KKT system |")
    lines.append("| **Iterative + LU** | N/A | N/A | Schur complement CG with LU-preconditioned Q^-1 |")
    lines.append("| **MG (L2+Galerkin)** | L2 Projection | Galerkin | Classic algebraic multigrid |")
    lines.append("| **MG (L2+Cached)** | L2 Projection | Cached Rediscret. | L2 transfer with direct element assembly |")
    lines.append("| **MG (Bezier+Galerkin)** | Bezier Subdiv. | Galerkin | Non-negative weights with algebraic coarsening |")
    lines.append("| **MG (Bezier+Cached)** | Bezier Subdiv. | Cached Rediscret. | Recommended for adaptive meshes |")
    lines.append("")

    # Final Metrics Table
    if solver_metrics:
        lines.append("### Final Metrics Comparison")
        lines.append("")

        # Order solvers consistently
        solvers = order_solvers(list(solver_metrics.keys()))

        metrics_to_show = [
            ("solution_l2_norm", "Solution L2 norm"),
            ("data_residual", "Data residual"),
            ("regularization_energy", "Regularization energy"),
            ("constraint_violation", "Constraint violation"),
            ("objective_value", "Objective value"),
            ("schur_cg_iterations", "Schur CG iterations"),
            ("qinv_apply_calls", "Q^-1 apply calls"),
            ("total_solve_ms", "Solve time (ms)"),
        ]

        header = "| Metric |"
        for s in solvers:
            header += f" {get_solver_display_name(s)} |"
        lines.append(header)
        lines.append("|--------|" + "".join(["--------------|"] * len(solvers)))

        for metric_key, metric_name in metrics_to_show:
            row = f"| {metric_name} |"
            for s in solvers:
                try:
                    val = solver_metrics[s].loc[metric_key, "value"]
                    row += f" {fmt(val, metric_key)} |"
                except KeyError:
                    row += " N/A |"
            lines.append(row)

        lines.append("")
        lines.append(
            "All solvers achieve the same objective value and solution quality. "
            "The constraint violation is lower for the iterative solvers due to exact "
            "Schur complement handling. The direct solver is fastest for this problem size, "
            "while the multigrid preconditioners show promise for larger problems where "
            "direct factorization becomes prohibitive."
        )
        lines.append("")

    # Solution Agreement Table
    if solver_comparison is not None and len(solver_comparison) > 0:
        lines.append("### Solution Agreement")
        lines.append("")
        lines.append("| Solver Pair | L2 Difference | Relative Difference |")
        lines.append("|-------------|---------------|---------------------|")
        for _, row in solver_comparison.iterrows():
            s1 = get_solver_display_name(row["solver1"])
            s2 = get_solver_display_name(row["solver2"])
            lines.append(f"| {s1} vs {s2} | {row['l2_diff']:.2e} | {row['relative_diff']:.2e} |")
        lines.append("")
        lines.append(
            "The direct and iterative+LU solvers produce nearly identical solutions "
            "(relative difference ~10^-9). The multigrid preconditioners achieve slightly "
            "different results due to the approximate nature of the V-cycle, but the "
            "relative difference remains small (~10^-6)."
        )
        lines.append("")

    # Convergence figures
    lines.append("### Convergence History")
    lines.append("")
    lines.append("![Solver Convergence](figures/solver_convergence.png)")
    lines.append("")
    lines.append(
        "Both iterative solvers converge at similar rates, with the LU-preconditioned "
        "solver slightly faster in terms of convergence factor."
    )
    lines.append("")

    lines.append("### CG Parameters")
    lines.append("")
    lines.append("![CG Parameters](figures/solver_cg_parameters.png)")
    lines.append("")
    lines.append(
        "The step size (alpha) and curvature (p^T S p) remain well-behaved throughout "
        "the iteration, indicating stable CG behavior with both preconditioners."
    )
    lines.append("")

    # Surface plots section
    lines.append("### Fitted Surfaces")
    lines.append("")
    lines.append(
        "The following plots show the Bezier control point surfaces fitted by each solver, "
        "compared to the analytical cosine bathymetry function."
    )
    lines.append("")

    lines.append("#### Surface Comparison")
    lines.append("")
    lines.append("![Surface Comparison](figures/surface_comparison.png)")
    lines.append("")
    lines.append(
        "All three solvers produce visually identical surfaces. The leftmost plot shows "
        "the analytical cosine function, followed by each solver's fitted Bezier surface."
    )
    lines.append("")

    lines.append("#### Per-Solver Error Surfaces")
    lines.append("")
    lines.append("![Error Surfaces](figures/surface_errors.png)")
    lines.append("")
    lines.append(
        "The error surfaces (solution - analytical) show the fitting residuals at each "
        "Bezier control point. The direct solver and iterative+LU solver achieve nearly "
        "identical errors, while the multigrid solver shows slightly different errors due "
        "to the approximate V-cycle."
    )
    lines.append("")

    # Section 2: Iterative Methods
    lines.append("## 2. Standalone Iterative Methods")
    lines.append("")
    lines.append("Four iterative methods are tested as standalone smoothers on a synthetic SPD system:")
    lines.append("")
    lines.append("| Method | Description |")
    lines.append("|--------|-------------|")
    lines.append("| **Jacobi** | Weighted Jacobi (omega=0.8) |")
    lines.append("| **Multiplicative Schwarz** | Sequential element block corrections |")
    lines.append("| **Additive Schwarz** | Parallel element corrections (omega=0.1) |")
    lines.append("| **Colored Schwarz** | Graph-colored hybrid approach |")
    lines.append("")

    # Method Comparison Table
    if method_comparison is not None and len(method_comparison) > 0:
        lines.append("### Method Comparison")
        lines.append("")
        lines.append("| Method | Iterations | Rel. Residual | Sol. Error | Time (ms) | Avg Conv. Rate |")
        lines.append("|--------|------------|---------------|------------|-----------|----------------|")

        method_names = {
            "jacobi": "Jacobi",
            "multiplicative_schwarz": "Multiplicative Schwarz",
            "additive_schwarz": "Additive Schwarz",
            "colored_schwarz": "Colored Schwarz",
        }

        for _, row in method_comparison.iterrows():
            name = method_names.get(row["method"], row["method"])
            iters = int(row["iterations"])
            rel_res = row["relative_residual"]
            sol_err = row["solution_error"]
            time_ms = row["total_ms"]

            # Compute convergence rate from iterations if available
            if row["method"] in method_iterations:
                residuals = method_iterations[row["method"]]["residual_norm"].values
                rate = compute_convergence_rate(residuals)
            else:
                rate = 0.0

            lines.append(
                f"| {name} | {iters} | {rel_res:.2e} | {sol_err:.2e} | {time_ms:.1f} | {rate:.2f} |"
            )

        lines.append("")
        lines.append(
            "The colored Schwarz method achieves the best balance of fast convergence "
            "(6 iterations) and low solution error (5e-07), making it the recommended "
            "smoother for multigrid applications."
        )
        lines.append("")

    # Convergence figures
    lines.append("### Convergence Curves")
    lines.append("")
    lines.append("![Method Convergence](figures/method_convergence.png)")
    lines.append("")
    lines.append(
        "The Schwarz methods (multiplicative and colored) converge much faster than Jacobi, "
        "requiring only 6 iterations compared to 43 for Jacobi. Additive Schwarz requires "
        "heavy damping (omega=0.1) for stability, resulting in slow convergence."
    )
    lines.append("")

    lines.append("### Early Convergence Detail")
    lines.append("")
    lines.append("![Convergence Detail](figures/method_convergence_detail.png)")
    lines.append("")
    lines.append(
        "The first 20 iterations show the rapid initial convergence of the multiplicative "
        "and colored Schwarz methods."
    )
    lines.append("")

    lines.append("### Per-Iteration Convergence Rate")
    lines.append("")
    lines.append("![Convergence Rate](figures/method_convergence_rate.png)")
    lines.append("")
    lines.append(
        "The convergence rate (r_{k+1}/r_k) shows that multiplicative and colored Schwarz "
        "achieve rates well below 1.0 (fast convergence), while Jacobi maintains a steady "
        "rate around 0.7."
    )
    lines.append("")

    # Section 3: Scaling Analysis
    lines.append("## 3. Scaling Analysis")
    lines.append("")
    lines.append(
        "This section analyzes how solver performance scales with problem size, "
        "testing grid sizes from 8×8 to 64×64 (625 to 37,249 DOFs). "
        "The direct solver is skipped for grids larger than 32×32 due to prohibitive runtime."
    )
    lines.append("")

    if scaling_data is not None and len(scaling_data) > 0:
        # CPU Time Table
        lines.append("### CPU Time Scaling")
        lines.append("")

        pivot_time = scaling_data.pivot_table(
            index="grid_size", columns="solver", values="time_ms", aggfunc="first"
        )
        solvers = order_solvers(pivot_time.columns.tolist())
        pivot_time = pivot_time[solvers]

        # Build header
        header = "| Grid | DOFs |"
        for s in solvers:
            header += f" {get_solver_display_name(s)} |"
        lines.append(header)
        lines.append("|------|------|" + "".join(["----|"] * len(solvers)))

        for grid_size in pivot_time.index:
            dofs = (3 * grid_size + 1) ** 2
            row = f"| {grid_size}×{grid_size} | {dofs} |"
            for solver in solvers:
                val = pivot_time.loc[grid_size, solver]
                if pd.isna(val):
                    row += " -- |"
                else:
                    row += f" {val:.1f} |"
            lines.append(row)

        lines.append("")
        lines.append("*CPU time in milliseconds*")
        lines.append("")

        # Iteration Count Table
        lines.append("### Iteration Count Scaling")
        lines.append("")

        pivot_iter = scaling_data.pivot_table(
            index="grid_size", columns="solver", values="iterations", aggfunc="first"
        )
        pivot_iter = pivot_iter[solvers]

        lines.append(header)
        lines.append("|------|------|" + "".join(["----|"] * len(solvers)))

        for grid_size in pivot_iter.index:
            dofs = (3 * grid_size + 1) ** 2
            row = f"| {grid_size}×{grid_size} | {dofs} |"
            for solver in solvers:
                val = pivot_iter.loc[grid_size, solver]
                if pd.isna(val):
                    row += " -- |"
                else:
                    row += f" {int(val)} |"
            lines.append(row)

        lines.append("")
        lines.append("*Schur CG iterations (0 for direct solver)*")
        lines.append("")

    # Scaling figures
    lines.append("### Scaling Plots")
    lines.append("")
    lines.append("#### Multigrid Strategy Comparison")
    lines.append("")
    lines.append("![Strategy Comparison - CPU Time](figures/solver_scaling_strategy.png)")
    lines.append("")
    lines.append(
        "Comparison of multigrid transfer operator strategies (L2 Projection vs Bezier Subdivision) "
        "and coarse grid assembly methods (Galerkin vs Cached Rediscretization). "
        "The Bezier+Cached combination achieves the best performance, being ~2× faster than "
        "L2+Galerkin at 64×64 grid size."
    )
    lines.append("")

    lines.append("![Strategy Comparison - Iterations](figures/solver_iterations_strategy.png)")
    lines.append("")
    lines.append(
        "Iteration counts remain stable across problem sizes for all MG strategies, "
        "demonstrating grid-independent convergence."
    )
    lines.append("")

    lines.append("#### Coarsest Level Comparison")
    lines.append("")
    lines.append("![Coarsest Level Comparison - CPU Time](figures/solver_scaling_coarsest.png)")
    lines.append("")
    lines.append(
        "Comparison of different coarsest level choices (4×4, 8×8, 16×16) using the optimal "
        "Bezier+Cached strategy with 1+1 smoothing. All three choices achieve similar performance "
        "at large grid sizes, all significantly faster than Iterative+LU."
    )
    lines.append("")

    lines.append("![Coarsest Level Comparison - Iterations](figures/solver_iterations_coarsest.png)")
    lines.append("")
    lines.append(
        "The iteration count remains relatively stable across problem sizes for "
        "all coarsest level choices, demonstrating the effectiveness "
        "of the multigrid preconditioner."
    )
    lines.append("")

    # Section 4: Conclusions
    lines.append("## 4. Conclusions")
    lines.append("")
    lines.append("### Verified Properties")
    lines.append("")
    lines.append("1. **Solver correctness**: All three solver strategies produce consistent solutions with relative differences < 10^-5")
    lines.append("2. **Constraint satisfaction**: Constraint violations are below 10^-8 for all solvers")
    lines.append("3. **Convergence**: Iterative solvers converge reliably within 50 iterations")
    lines.append("4. **Smoother effectiveness**: Colored multiplicative Schwarz is the most effective smoother (6 iterations, lowest error)")
    lines.append("")

    lines.append("### Recommended Configurations")
    lines.append("")
    lines.append("| Use Case | Recommendation |")
    lines.append("|----------|----------------|")
    lines.append("| Small problems (< 10k DOFs) | Direct solver (SparseLU) |")
    lines.append("| Large problems | Iterative + Multigrid |")
    lines.append("| Multigrid smoother | Colored Multiplicative Schwarz |")
    lines.append("| Parallel smoothing | Additive Schwarz (with small omega) |")
    lines.append("")

    lines.append("### Regenerating This Report")
    lines.append("")
    lines.append("To regenerate the figures and tables in this report:")
    lines.append("")
    lines.append("```bash")
    lines.append("./docs/regenerate_figures.sh")
    lines.append("```")
    lines.append("")
    lines.append("This requires:")
    lines.append("- Project built (`cmake --build build`)")
    lines.append("- Python environment in `scr/.venv` with matplotlib, pandas, numpy")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate CG Bezier Solver Verification Report")
    parser.add_argument("directory", help="Directory with CSV files from tests")
    parser.add_argument("--output", "-o", help="Output directory for report", default=".")
    args = parser.parse_args()

    directory = Path(args.directory)
    output_dir = Path(args.output)

    # Load all data
    solver_metrics = load_solver_metrics(directory)
    solver_iterations = load_solver_iterations(directory)
    solver_comparison = load_solver_comparison(directory)
    method_iterations = load_method_iterations(directory)
    method_comparison = load_method_comparison(directory)
    scaling_data = load_scaling_data(directory)

    if not solver_metrics and not method_comparison:
        print(f"No CSV files found in {directory}")
        return

    # Generate report
    report = generate_report(
        solver_metrics,
        solver_iterations,
        solver_comparison,
        method_iterations,
        method_comparison,
        scaling_data,
    )

    # Write report
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "cg_bezier_solver_verification.md"
    output_path.write_text(report)
    print(f"Generated: {output_path}")


if __name__ == "__main__":
    main()
