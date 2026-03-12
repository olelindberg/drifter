#!/bin/bash
# Regenerate verification figures and report for CG Bezier solver
#
# Usage:
#   ./docs/regenerate_figures.sh
#
# Prerequisites:
#   - Project built (cmake --build build)
#   - Python environment in scr/.venv with matplotlib, pandas, numpy

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FIGURES_DIR="$SCRIPT_DIR/figures"
SCR_DIR="$PROJECT_ROOT/scr"

echo "=== CG Bezier Solver Verification Report Generation ==="
echo "Project root: $PROJECT_ROOT"
echo "Figures dir:  $FIGURES_DIR"

# Ensure figures directory exists
mkdir -p "$FIGURES_DIR"

# Set library path for GDAL and other custom libs
export LD_LIBRARY_PATH=/home/ole/.local/lib:$LD_LIBRARY_PATH

# Run the solver comparison test (generates CSV files to /tmp)
echo ""
echo "=== Running solver comparison test ==="
"$PROJECT_ROOT/build/tests/drifter_integration_tests" --gtest_filter="*AllSolversComparison*" || {
    echo "Warning: Solver comparison test failed or not available"
}

# Run the iterative methods test (generates CSV files to /tmp)
echo ""
echo "=== Running iterative methods test ==="
"$PROJECT_ROOT/build/tests/drifter_integration_tests" --gtest_filter="*AllMethodsComparison*" || {
    echo "Warning: Iterative methods test failed or not available"
}

# Run the scaling test (generates solver_scaling.csv to /tmp)
echo ""
echo "=== Running solver scaling test ==="
"$PROJECT_ROOT/build/tests/drifter_integration_tests" --gtest_filter="*AllSolversScalingTest*" || {
    echo "Warning: Solver scaling test failed or not available"
}

# Generate solver verification figures
echo ""
echo "=== Generating solver verification figures ==="
cd "$SCR_DIR"
uv run solver_verification.py /tmp --output "$FIGURES_DIR"

# Generate iterative methods figures
echo ""
echo "=== Generating iterative methods figures ==="
uv run iterative_methods.py /tmp --output "$FIGURES_DIR"

# Generate surface plots
echo ""
echo "=== Generating surface plots ==="
uv run surface_plots.py /tmp --output "$FIGURES_DIR"

# Generate scaling plots
echo ""
echo "=== Generating scaling plots ==="
uv run solver_scaling.py /tmp --output "$FIGURES_DIR"

# Generate the markdown report from CSV data
echo ""
echo "=== Generating verification report ==="
uv run generate_report.py /tmp --output "$SCRIPT_DIR"

echo ""
echo "=== Done ==="
echo "Figures saved to: $FIGURES_DIR"
echo "Report saved to:  $SCRIPT_DIR/cg_bezier_solver_verification.md"
ls -la "$FIGURES_DIR"
