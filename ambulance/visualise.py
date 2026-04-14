"""
Visualisation helpers for the ambulance routing model.

All functions accept an optional save_path argument.
  - If save_path is given, the figure is saved to that file and closed.
  - If save_path is None, plt.show() is called instead.
"""

import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; works headless and in WSL
import matplotlib.pyplot as plt
import numpy as np

from .demand import calculate_b_gaussian, calculate_b_centroid


def _finish(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_simulation(
    ambulance_posts,
    assignments_history,
    sample_points=None,
    intensities_history=None,
    save_path=None,
):
    """
    Plot assignment maps for each timestep in a 2-column grid.

    If sample_points and intensities_history are given, draw the demand
    heatmap as background scatter.
    """
    num_timesteps = len(assignments_history)
    num_cols = 2
    num_rows = (num_timesteps + 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10 * num_cols, 8 * num_rows))
    axes = axes.flatten()

    xs = [p[0] for p in sample_points] if sample_points else []
    ys = [p[1] for p in sample_points] if sample_points else []

    for t, assignments in enumerate(assignments_history):
        ax = axes[t]

        if sample_points and intensities_history:
            ax.scatter(xs, ys, c=intensities_history[t], cmap="YlOrRd",
                       s=4, alpha=0.5, zorder=1)

        for p_idx, post in enumerate(ambulance_posts):
            ax.scatter(post[0], post[1], c="blue", marker="s", s=80, zorder=5,
                       label="Post" if p_idx == 0 else "")
            ax.text(post[0] + 0.3, post[1] + 0.3, f"P{p_idx}", color="blue", fontsize=7)

        for amb_idx, post_idx in sorted(assignments):
            post = ambulance_posts[post_idx]
            ax.scatter(post[0], post[1], c="#00AA00", marker="^", s=220,
                       zorder=10, edgecolors="darkgreen", linewidth=1.5,
                       label="Ambulance" if amb_idx == 0 else "")
            ax.text(post[0] - 0.6, post[1] - 0.6, f"A{amb_idx}",
                    color="white", fontsize=9, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="darkgreen", alpha=0.8),
                    zorder=11)

        ax.set_title(f"Timestep {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(True, alpha=0.3)
        if t == 0:
            ax.legend(fontsize=8)

    for idx in range(num_timesteps, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    _finish(fig, save_path)


def visualise_gaussian_states(
    sample_points,
    ambulance_posts,
    map_bounds,
    n_states=6,
    save_path=None,
):
    """Plot raw Gaussian intensity for n_states demand configurations."""
    num_cols = 2
    num_rows = (n_states + 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 6 * num_rows))
    axes = axes.flatten()

    xs = [p[0] for p in sample_points]
    ys = [p[1] for p in sample_points]

    titles = [
        "State 0: 2 Gaussians – opposite corners",
        "State 1: 3 Gaussians – equilateral triangle",
        "State 2: 2 Gaussians – randomised (seed 2024)",
        "State 3: 3 Gaussians – randomised (seed 2025)",
    ]

    for t in range(n_states):
        intensities = calculate_b_gaussian(ambulance_posts, t, sample_points, map_bounds)
        ax = axes[t]
        sc = ax.scatter(xs, ys, c=intensities, cmap="YlOrRd", s=6, alpha=0.8)
        for idx, post in enumerate(ambulance_posts):
            ax.scatter(post[0], post[1], c="blue", marker="s", s=60, zorder=5)
            ax.text(post[0] + 0.3, post[1] + 0.3, f"P{idx}", color="blue", fontsize=6)
        plt.colorbar(sc, ax=ax, label="Intensity")
        ax.set_title(titles[t % 4], fontsize=10)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    for idx in range(n_states, len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    _finish(fig, save_path)


def visualise_centroid_analysis(
    sample_points,
    ambulance_posts,
    map_bounds,
    num_ambulances=3,
    n_states=4,
    save_path=None,
):
    """
    4×3 grid showing, for each demand state:
      col 0 – raw Gaussian intensity
      col 1 – k-means cluster membership + centroids
      col 2 – centroid-adjusted benefit
    """
    fig, axes = plt.subplots(n_states, 3, figsize=(20, 6 * n_states))

    xs = [p[0] for p in sample_points]
    ys = [p[1] for p in sample_points]

    for row, t in enumerate(range(n_states)):
        raw_b = calculate_b_gaussian(ambulance_posts, t, sample_points, map_bounds)
        b_adj, centroids, labels = calculate_b_centroid(
            ambulance_posts, t, sample_points, map_bounds, num_ambulances
        )

        # --- col 0: raw intensity ---
        ax0 = axes[row, 0]
        sc0 = ax0.scatter(xs, ys, c=raw_b, cmap="YlOrRd", s=4, alpha=0.8)
        for post in ambulance_posts:
            ax0.scatter(post[0], post[1], c="blue", marker="s", s=60, zorder=5)
        plt.colorbar(sc0, ax=ax0, label="Intensity")
        ax0.set_title(f"State {t}: Raw Intensity", fontsize=10)
        ax0.set_aspect("equal")
        ax0.grid(True, alpha=0.3)

        # --- col 1: clusters ---
        ax1 = axes[row, 1]
        ax1.scatter(xs, ys, c=labels, cmap="tab10", s=4, alpha=0.4)
        for post in ambulance_posts:
            ax1.scatter(post[0], post[1], c="blue", marker="s", s=60, zorder=5)
        for c_idx, cent in enumerate(centroids):
            ax1.scatter(cent[0], cent[1], c="red", marker="X", s=200,
                        zorder=10, edgecolors="black", linewidth=1.5)
            ax1.text(cent[0] + 0.5, cent[1] + 0.5, f"C{c_idx}",
                     color="red", fontsize=8, weight="bold")
        ax1.set_title(f"State {t}: K-means Clusters (K={num_ambulances})", fontsize=10)
        ax1.set_aspect("equal")
        ax1.grid(True, alpha=0.3)

        # --- col 2: adjusted benefit ---
        ax2 = axes[row, 2]
        sc2 = ax2.scatter(xs, ys, c=b_adj, cmap="YlOrRd", s=4, alpha=0.8)
        for post in ambulance_posts:
            ax2.scatter(post[0], post[1], c="blue", marker="s", s=60, zorder=5)
        for cent in centroids:
            ax2.scatter(cent[0], cent[1], c="red", marker="X", s=200,
                        zorder=10, edgecolors="black", linewidth=1.5)
        plt.colorbar(sc2, ax=ax2, label="Adjusted Benefit")
        ax2.set_title(f"State {t}: Centroid-Adjusted Benefit", fontsize=10)
        ax2.set_aspect("equal")
        ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Centroid-Based Benefit: Raw Intensity → Clusters → Adjusted Benefit",
        fontsize=14,
        y=1.005,
    )
    plt.tight_layout()
    _finish(fig, save_path)


def print_comparison(assignment_history_gauss, assignment_history_centroid):
    """Print a side-by-side comparison of Gaussian vs centroid assignments."""
    n = min(len(assignment_history_gauss), len(assignment_history_centroid))
    print("=== Assignment Comparison ===")
    print(f"{'Timestep':<10} {'Gaussian':>30} {'Centroid':>30}")
    print("-" * 72)
    for t in range(n):
        g_posts = [j for _, j in sorted(assignment_history_gauss[t])]
        c_posts = [j for _, j in sorted(assignment_history_centroid[t])]
        marker = " *" if g_posts != c_posts else ""
        print(f"t={t:<8} {str(g_posts):>30} {str(c_posts):>30}{marker}")
    print("\n* = assignments differ between methods")
