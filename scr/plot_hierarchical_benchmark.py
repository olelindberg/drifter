#!/usr/bin/env python3
"""
Plot hierarchical ordering benchmark results from real workloads.

Reads the benchmark markdown file and generates performance charts.

Usage:
    python plot_hierarchical_benchmark.py
    python plot_hierarchical_benchmark.py --output docs/hierarchical_benchmark.png
"""

import argparse
import re
import os
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_markdown_table(md_file: str) -> dict:
    """Parse benchmark results from markdown file."""
    with open(md_file, 'r') as f:
        content = f.read()

    results = {'standard': [], 'hierarchical': []}

    # Find the results table
    # Format: | Iterations | Config | Elements | DOFs | Time (ms) | Speedup |
    table_pattern = r'\| (\d+) \| (standard|hierarchical) \| (\d+) \| ~(\d+) \| ([\d.]+) \| ([\d.]+)x \|'
    matches = re.findall(table_pattern, content)

    for match in matches:
        iterations = int(match[0])
        config = match[1]
        elements = int(match[2])
        dofs = int(match[3])
        time_ms = float(match[4])
        speedup = float(match[5])

        results[config].append({
            'iterations': iterations,
            'elements': elements,
            'dofs': dofs,
            'time_ms': time_ms,
            'speedup': speedup
        })

    return results


def plot_scaling(results: dict, output_file: str | None = None):
    """Create scaling plot showing time vs elements."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Colors
    colors = {'standard': '#4C72B0', 'hierarchical': '#55A868'}
    markers = {'standard': 'o', 'hierarchical': 's'}

    # Plot 1: Time vs Elements
    for config, data in results.items():
        if not data:
            continue
        elements = [d['elements'] for d in data]
        times = [d['time_ms'] / 1000 for d in data]  # Convert to seconds
        ax1.plot(elements, times, marker=markers[config], color=colors[config],
                 label=config.capitalize(), linewidth=2, markersize=8)

    ax1.set_xlabel('Number of Elements')
    ax1.set_ylabel('Total Time (seconds)')
    ax1.set_title('Solver Scaling: Time vs Mesh Size')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # Plot 2: Speedup vs Elements
    if results['standard'] and results['hierarchical']:
        std_data = {d['iterations']: d for d in results['standard']}
        hier_data = {d['iterations']: d for d in results['hierarchical']}

        common_iters = sorted(set(std_data.keys()) & set(hier_data.keys()))
        elements = [std_data[i]['elements'] for i in common_iters]
        speedups = [std_data[i]['time_ms'] / hier_data[i]['time_ms'] for i in common_iters]

        ax2.bar(range(len(elements)), speedups, color=colors['hierarchical'], alpha=0.7, edgecolor='black')
        ax2.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='No speedup (1.0x)')
        ax2.set_xticks(range(len(elements)))
        ax2.set_xticklabels([str(e) for e in elements])
        ax2.set_xlabel('Number of Elements')
        ax2.set_ylabel('Speedup (hierarchical vs standard)')
        ax2.set_title('Hierarchical Ordering Speedup')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (elem, sp) in enumerate(zip(elements, speedups)):
            ax2.annotate(f'{sp:.2f}x', xy=(i, sp), ha='center', va='bottom' if sp >= 1 else 'top',
                        fontsize=10, fontweight='bold')

    plt.suptitle('Hierarchical DOF Ordering Performance (Real Workloads)', fontsize=12, y=1.02)
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()


def plot_time_breakdown(results: dict, output_file: str | None = None):
    """Create bar chart comparing configurations at each iteration count."""

    if not results['standard']:
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get iteration counts
    iterations = sorted(set(d['iterations'] for d in results['standard']))
    n_iters = len(iterations)

    # Bar width and positions
    width = 0.35
    x = np.arange(n_iters)

    # Get times for each config
    std_times = []
    hier_times = []
    elements = []

    std_dict = {d['iterations']: d for d in results['standard']}
    hier_dict = {d['iterations']: d for d in results['hierarchical']}

    for it in iterations:
        std_times.append(std_dict.get(it, {}).get('time_ms', 0) / 1000)
        hier_times.append(hier_dict.get(it, {}).get('time_ms', 0) / 1000)
        elements.append(std_dict.get(it, {}).get('elements', 0))

    # Create bars
    bars1 = ax.bar(x - width/2, std_times, width, label='Standard', color='#4C72B0')
    bars2 = ax.bar(x + width/2, hier_times, width, label='Hierarchical', color='#55A868')

    # Labels
    ax.set_xlabel('Adaptive Iterations')
    ax.set_ylabel('Total Time (seconds)')
    ax.set_title('Solver Performance: Standard vs Hierarchical DOF Ordering')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{it}\n({elem} elem)' for it, elem in zip(iterations, elements)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}s',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)

    plt.tight_layout()

    if output_file:
        base, ext = os.path.splitext(output_file)
        time_file = f"{base}_time{ext}"
        plt.savefig(time_file, dpi=150, bbox_inches='tight')
        print(f"Saved time breakdown plot to {time_file}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot hierarchical ordering benchmark results"
    )
    parser.add_argument(
        "--input", "-i",
        default="docs/hierarchical_ordering_benchmark.md",
        help="Input markdown file (default: docs/hierarchical_ordering_benchmark.md)"
    )
    parser.add_argument(
        "--output", "-o",
        help="Output file (default: show interactively)"
    )

    args = parser.parse_args()

    # Find input file
    if not os.path.isabs(args.input):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        args.input = os.path.join(project_root, args.input)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Parse results
    results = parse_markdown_table(args.input)

    print(f"Loaded benchmark results:")
    for config, data in results.items():
        print(f"  {config}: {len(data)} data points")
        for d in data:
            print(f"    {d['iterations']} iters: {d['elements']} elements, {d['time_ms']:.1f}ms")

    # Plot
    plot_scaling(results, args.output)
    plot_time_breakdown(results, args.output)


if __name__ == "__main__":
    main()
