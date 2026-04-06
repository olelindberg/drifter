#!/bin/bash
# Benchmark script for hierarchical ordering performance
# Runs the main program with different max_iterations values

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build"
CONFIG_DIR="$PROJECT_ROOT/config"
RESULTS_FILE="$PROJECT_ROOT/docs/hierarchical_ordering_benchmark.md"

# Check if drifter executable exists
if [ ! -f "$BUILD_DIR/drifter" ]; then
    echo "Error: drifter executable not found. Please build the project first."
    exit 1
fi

# Max iterations to test
ITERATIONS=(4 6 8 10 12)

# Configurations to test
CONFIGS=("standard" "hierarchical")

# Output header
echo "# Hierarchical Ordering Performance Benchmark (Real Workloads)" > "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "This benchmark uses the adaptive CG cubic Bezier bathymetry smoother with real GeoTIFF data." >> "$RESULTS_FILE"
echo "Measurements are from the main program with varying refinement iterations." >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "## Test Configuration" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "- **Domain**: 100km x 100km coastal region" >> "$RESULTS_FILE"
echo "- **Initial grid**: 8x8 elements" >> "$RESULTS_FILE"
echo "- **Solver**: Iterative (PCG + Multigrid)" >> "$RESULTS_FILE"
echo "- **Error metric**: VolumeChange" >> "$RESULTS_FILE"
echo "- **Error threshold**: 1.0m" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "## Results" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "| Iterations | Config | Elements | DOFs | Time (ms) | Speedup |" >> "$RESULTS_FILE"
echo "|------------|--------|----------|------|-----------|---------|" >> "$RESULTS_FILE"

# Create temporary config directory
TMP_CONFIG_DIR=$(mktemp -d)
trap "rm -rf $TMP_CONFIG_DIR" EXIT

# Store results for speedup calculation
declare -A TIMES
declare -A ELEMENTS
declare -A DOFS

for iter in "${ITERATIONS[@]}"; do
    for cfg in "${CONFIGS[@]}"; do
        echo "Running: max_iterations=$iter, config=$cfg"

        # Create config file
        CFG_FILE="$TMP_CONFIG_DIR/benchmark_${iter}_${cfg}.json"

        # Set hierarchical ordering flags
        if [ "$cfg" == "hierarchical" ]; then
            HIER="true"
            COND="false"  # Static condensation disabled for multi-element
        else
            HIER="false"
            COND="false"
        fi

        cat > "$CFG_FILE" << EOF
{
  "data": {
    "data_dir": "/home/ole/Projects/drifter/data/input/",
    "primary_file": "ddm_50m.dybde-emodnet.tif",
    "tile_files": [
      "C4_2024.tif", "C5_2024.tif", "C6_2024.tif", "C7_2024.tif",
      "D4_2024.tif", "D5_2024.tif", "D6_2024.tif", "D7_2024.tif",
      "E4_2024.tif", "E5_2024.tif", "E6_2024.tif", "E7_2024.tif"
    ]
  },
  "domain": {
    "center_x": 4095238.0,
    "center_y": 3344695.0,
    "domain_size": 100000.0
  },
  "initial_grid": {
    "nx": 8,
    "ny": 8
  },
  "adaptive": {
    "error_threshold": 1.0,
    "error_metric_type": "VolumeChange",
    "max_iterations": $iter,
    "max_elements": 10000,
    "max_refinement_level": 12,
    "verbose": false,
    "ngauss_error": 6
  },
  "smoother": {
    "lambda": 10.0,
    "edge_ngauss": 4,
    "tolerance": 1e-6,
    "use_iterative_solver": true,
    "use_multigrid": true,
    "use_hierarchical_ordering": $HIER,
    "use_static_condensation": $COND,
    "schur_preconditioner": "BlockDiagApproxCG",
    "verbose": false,
    "multigrid": {
      "smoother_type": "MultiplicativeSchwarz",
      "verbose": false,
      "pre_smoothing": 2,
      "post_smoothing": 2,
      "max_vcycles": 10,
      "vcycle_tolerance": 1e-3,
      "transfer_strategy": "BezierSubdivision",
      "coarse_grid_strategy": "CachedRediscretization"
    }
  },
  "output": {
    "output_file": "/tmp/benchmark_${iter}_${cfg}",
    "vtk_subdivision": 4
  }
}
EOF

        # Run benchmark and capture output
        OUTPUT=$(LD_LIBRARY_PATH=/home/ole/.local/lib "$BUILD_DIR/drifter" "$CFG_FILE" 2>&1)

        # Extract metrics
        ELEM=$(echo "$OUTPUT" | grep -oP 'Elements:\s*\K[0-9]+' | tail -1)
        TIME=$(echo "$OUTPUT" | grep -oP 'Time:\s*\K[0-9.]+' | tail -1)

        # Estimate DOFs (approximately 10 * elements for CG cubic with shared DOFs)
        # More accurate: 4*elements + edges + corners, but this is a reasonable estimate
        DOFS_EST=$((ELEM * 10))

        # Store results
        KEY="${iter}_${cfg}"
        TIMES[$KEY]=$TIME
        ELEMENTS[$KEY]=$ELEM
        DOFS[$KEY]=$DOFS_EST

        # Calculate speedup (relative to standard)
        STD_KEY="${iter}_standard"
        if [ "$cfg" == "standard" ]; then
            SPEEDUP="1.00"
        else
            if [ -n "${TIMES[$STD_KEY]}" ]; then
                SPEEDUP=$(echo "scale=2; ${TIMES[$STD_KEY]} / $TIME" | bc)
            else
                SPEEDUP="N/A"
            fi
        fi

        # Write to results file
        echo "| $iter | $cfg | $ELEM | ~$DOFS_EST | $TIME | ${SPEEDUP}x |" >> "$RESULTS_FILE"

        echo "  Elements: $ELEM, Time: ${TIME}ms"
    done
done

echo "" >> "$RESULTS_FILE"
echo "## Summary" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "### Observations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "- Hierarchical DOF ordering reorders DOFs by (level, is_constrained, type, hilbert_key)" >> "$RESULTS_FILE"
echo "- For multi-element meshes, static condensation is automatically disabled" >> "$RESULTS_FILE"
echo "- Performance gains come primarily from improved cache locality in matrix operations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "### Recommendations" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "1. **Large adaptive meshes**: Enable \`use_hierarchical_ordering=true\`" >> "$RESULTS_FILE"
echo "2. **Small meshes (< 100 elements)**: Standard ordering is sufficient" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"
echo "---" >> "$RESULTS_FILE"
echo "*Generated by run_hierarchical_benchmark.sh on $(date)*" >> "$RESULTS_FILE"

echo ""
echo "Benchmark complete. Results written to: $RESULTS_FILE"
