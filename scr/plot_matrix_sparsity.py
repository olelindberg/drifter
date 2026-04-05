#!/usr/bin/env python3
"""
Plot sparsity patterns of CG Bezier smoother matrices for multiple mesh sizes.

Usage:
    cd scr && uv run plot_matrix_sparsity.py /tmp/drifter_matrix_sparsity
    cd scr && uv run plot_matrix_sparsity.py /tmp/drifter_matrix_sparsity --output /tmp/out
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import mmread

MATRIX_DEFS = [
    ("H.mtx", "H (thin plate hessian)"),
    ("BtWB.mtx", r"$B^T W B$ (data fitting)"),
    ("Q.mtx", "Q (system matrix)"),
    ("A_edge.mtx", r"$A_{edge}$ (C$^1$ constraints)"),
    ("AAT.mtx", r"$A A^T$"),
    ("S.mtx", r"$A Q^{-1} A^T$ (Schur)"),
    ("Q_blkdiag.mtx", r"blockdiag($Q$)"),
    ("Q_blkdiag_inv.mtx", r"blockdiag($Q$)$^{-1}$"),
    ("MS_diag.mtx", r"$M_S$ diag approx"),
    ("MS_block.mtx", r"$M_S$ block-diag approx"),
]


def load_dof_map(path):
    """Load DOF map: returns (dof_elem, num_elements).

    dof_elem[i] = element index for interior DOFs, -1 for shared DOFs.
    """
    with open(path) as f:
        header = f.readline().split()
        num_dofs = int(header[0])
        num_elems = int(header[1])
        dof_elem = np.full(num_dofs, -1, dtype=int)
        for line in f:
            parts = line.split()
            dof_elem[int(parts[0])] = int(parts[1])
    return dof_elem, num_elems


def build_element_colormap(num_elems):
    """Build a list of distinct colors for elements."""
    base_cmap = plt.colormaps["tab20"]._resample(max(num_elems, 1))
    return [base_cmap(i) for i in range(num_elems)]


def plot_colored_spy(ax, mat, dof_elem_row, dof_elem_col, elem_colors, markersize):
    """Plot spy with entries colored by element ownership.

    An entry (i,j) is colored by element e if BOTH row DOF i and column DOF j
    are interior to element e. Otherwise it is black (shared coupling).
    """
    coo = mat.tocoo()
    rows, cols = coo.row, coo.col

    row_elem = dof_elem_row[rows]
    col_elem = dof_elem_col[cols]

    same_interior = (row_elem == col_elem) & (row_elem >= 0)

    shared = ~same_interior
    if np.any(shared):
        ax.plot(cols[shared], rows[shared], ".", color="black",
                markersize=markersize, rasterized=True)

    if np.any(same_interior):
        for elem_id in np.unique(row_elem[same_interior]):
            mask = same_interior & (row_elem == elem_id)
            color = elem_colors[elem_id % len(elem_colors)]
            ax.plot(cols[mask], rows[mask], ".", color=color,
                    markersize=markersize, rasterized=True)

    ax.set_xlim(-0.5, mat.shape[1] - 0.5)
    ax.set_ylim(mat.shape[0] - 0.5, -0.5)
    ax.set_aspect("equal")


def plot_one_mesh(subdir, output_path, markersize):
    """Generate a single figure for one mesh size."""
    mesh_label = subdir.name

    dof_map_path = subdir / "dof_map.txt"
    if dof_map_path.exists():
        dof_elem, num_elems = load_dof_map(dof_map_path)
        elem_colors = build_element_colormap(num_elems)
    else:
        dof_elem, num_elems, elem_colors = None, 0, []

    # Find which matrices exist
    available = [(f, label) for f, label in MATRIX_DEFS if (subdir / f).exists()]
    # Add "empty" placeholders for missing files that have nnz=0
    for f, label in MATRIX_DEFS:
        path = subdir / f
        if not path.exists() and (f, label) not in available:
            available.append((f, label))

    n = len(MATRIX_DEFS)
    ncols = 4
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_2d(axes)

    for idx, (filename, matrix_label) in enumerate(MATRIX_DEFS):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        path = subdir / filename

        if not path.exists():
            ax.text(0.5, 0.5, "empty", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(matrix_label, fontsize=12)
            continue

        mat = mmread(str(path))

        if mat.nnz == 0:
            ax.text(0.5, 0.5, "empty", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
        elif dof_elem is not None:
            n_row_dofs = mat.shape[0]
            n_col_dofs = mat.shape[1]

            col_map = dof_elem[:n_col_dofs] if n_col_dofs <= len(dof_elem) else np.full(n_col_dofs, -1)

            if n_row_dofs == len(dof_elem):
                row_map = dof_elem
            else:
                row_map = np.full(n_row_dofs, -1)

            plot_colored_spy(ax, mat, row_map, col_map, elem_colors, markersize)
        else:
            ax.spy(mat, markersize=markersize)

        ax.set_title(
            f"{matrix_label}\n{mat.shape[0]}x{mat.shape[1]}, nnz={mat.nnz}",
            fontsize=11,
        )
        ax.set_xlabel("column")
        ax.set_ylabel("row")

    # Hide unused axes
    for idx in range(len(MATRIX_DEFS), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"CG Cubic Bezier Smoother — {mesh_label} elements",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot sparsity patterns of CG Bezier smoother matrices"
    )
    parser.add_argument("input_dir", type=Path, help="Directory containing NxN/ subdirs with .mtx files")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory for PNGs (default: input_dir)",
    )
    parser.add_argument(
        "--markersize", type=float, default=2.0, help="Marker size for spy plot"
    )
    args = parser.parse_args()

    output_dir = args.output or args.input_dir

    def sort_key(d):
        """Sort by mesh size: NxN dirs first, then center_graded_LN dirs."""
        if "x" in d.name:
            return (0, int(d.name.split("x")[0]))
        elif d.name.startswith("center_graded_L"):
            return (1, int(d.name.split("L")[1]))
        return (2, 0)

    subdirs = sorted(
        [d for d in args.input_dir.iterdir()
         if d.is_dir() and ("x" in d.name or d.name.startswith("center_graded_"))],
        key=sort_key,
    )

    if not subdirs:
        print(f"No mesh subdirectories found in {args.input_dir}")
        return

    print(f"Generating {len(subdirs)} plots:")
    for subdir in subdirs:
        out_path = output_dir / f"sparsity_{subdir.name}.png"
        plot_one_mesh(subdir, out_path, args.markersize)


if __name__ == "__main__":
    main()
