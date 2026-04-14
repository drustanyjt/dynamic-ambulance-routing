"""
Dynamic Ambulance Routing — CLI entry point

Commands
--------
simulate   Run the receding horizon simulation
compare    Run both Gaussian and centroid methods side-by-side
visualise  Plot demand field or centroid analysis
"""

import argparse
import math
import os

from ambulance import (
    make_grid_points,
    calculate_b_gaussian,
    calculate_b_centroid,
    make_nearest_coverage,
    run_simulation,
    polygon_vertices,
)
from ambulance.visualise import (
    plot_simulation,
    visualise_gaussian_states,
    visualise_centroid_analysis,
    print_comparison,
)


# ── Shared setup ─────────────────────────────────────────────────────────────

def build_scenario(num_posts=10, num_ambulances=3, grid_step=1):
    ambulance_posts = polygon_vertices(num_posts, center=(0, 0), radius=15)
    map_bounds = (-22, 22, -22, 22)
    sample_points = make_grid_points(map_bounds, grid_step)
    J_k = make_nearest_coverage(sample_points, ambulance_posts)
    return ambulance_posts, map_bounds, sample_points, J_k


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_simulate(args):
    ambulance_posts, map_bounds, sample_points, J_k = build_scenario(
        num_ambulances=args.ambulances
    )

    if args.method == "gaussian":
        def calculate_b_fn(assignments, posts, t):
            return calculate_b_gaussian(posts, t, sample_points, map_bounds)
    else:
        def calculate_b_fn(assignments, posts, t):
            b_adj, _, _ = calculate_b_centroid(
                posts, t, sample_points, map_bounds, args.ambulances
            )
            return b_adj

    history = run_simulation(
        num_timesteps=args.timesteps,
        num_ambulances=args.ambulances,
        ambulance_posts=ambulance_posts,
        call_hotspots=sample_points,
        J_k=J_k,
        calculate_b_fn=calculate_b_fn,
        verbose=True,
    )

    intensities = [
        calculate_b_gaussian(ambulance_posts, t, sample_points, map_bounds)
        for t in range(args.timesteps)
    ]

    save_path = os.path.join(args.save_dir, "simulation.png") if args.save_dir else None
    plot_simulation(
        ambulance_posts,
        history,
        sample_points=sample_points,
        intensities_history=intensities,
        save_path=save_path,
    )


def cmd_compare(args):
    ambulance_posts, map_bounds, sample_points, J_k = build_scenario(
        num_ambulances=args.ambulances
    )

    def b_gaussian(assignments, posts, t):
        return calculate_b_gaussian(posts, t, sample_points, map_bounds)

    def b_centroid(assignments, posts, t):
        b_adj, _, _ = calculate_b_centroid(
            posts, t, sample_points, map_bounds, args.ambulances
        )
        return b_adj

    print("Running Gaussian simulation...")
    history_gauss = run_simulation(
        num_timesteps=args.timesteps,
        num_ambulances=args.ambulances,
        ambulance_posts=ambulance_posts,
        call_hotspots=sample_points,
        J_k=J_k,
        calculate_b_fn=b_gaussian,
        verbose=False,
    )

    print("Running centroid simulation...")
    history_centroid = run_simulation(
        num_timesteps=args.timesteps,
        num_ambulances=args.ambulances,
        ambulance_posts=ambulance_posts,
        call_hotspots=sample_points,
        J_k=J_k,
        calculate_b_fn=b_centroid,
        verbose=False,
    )

    print_comparison(history_gauss, history_centroid)

    if args.save_dir:
        intensities = [
            calculate_b_gaussian(ambulance_posts, t, sample_points, map_bounds)
            for t in range(args.timesteps)
        ]
        plot_simulation(
            ambulance_posts, history_gauss,
            sample_points=sample_points,
            intensities_history=intensities,
            save_path=os.path.join(args.save_dir, "simulation_gaussian.png"),
        )
        plot_simulation(
            ambulance_posts, history_centroid,
            sample_points=sample_points,
            intensities_history=intensities,
            save_path=os.path.join(args.save_dir, "simulation_centroid.png"),
        )


def cmd_visualise(args):
    ambulance_posts, map_bounds, sample_points, _ = build_scenario()

    if args.plot == "demand":
        save_path = os.path.join(args.save_dir, "demand_states.png") if args.save_dir else None
        visualise_gaussian_states(
            sample_points, ambulance_posts, map_bounds,
            n_states=args.states,
            save_path=save_path,
        )
    elif args.plot == "centroid":
        save_path = os.path.join(args.save_dir, "centroid_analysis.png") if args.save_dir else None
        visualise_centroid_analysis(
            sample_points, ambulance_posts, map_bounds,
            num_ambulances=args.ambulances,
            n_states=args.states,
            save_path=save_path,
        )


# ── Argument parsing ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Dynamic ambulance routing — receding horizon MILP",
    )
    parser.add_argument(
        "--save-dir", metavar="DIR", default=None,
        help="Save figures to this directory instead of displaying them",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # simulate
    p_sim = sub.add_parser("simulate", help="Run receding horizon simulation")
    p_sim.add_argument("--method", choices=["gaussian", "centroid"], default="gaussian")
    p_sim.add_argument("--timesteps", type=int, default=6)
    p_sim.add_argument("--ambulances", type=int, default=3)
    p_sim.set_defaults(func=cmd_simulate)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare Gaussian vs centroid assignments")
    p_cmp.add_argument("--timesteps", type=int, default=6)
    p_cmp.add_argument("--ambulances", type=int, default=3)
    p_cmp.set_defaults(func=cmd_compare)

    # visualise
    p_vis = sub.add_parser("visualise", help="Plot demand field or centroid analysis")
    p_vis.add_argument("--plot", choices=["demand", "centroid"], default="demand")
    p_vis.add_argument("--states", type=int, default=4)
    p_vis.add_argument("--ambulances", type=int, default=3)
    p_vis.set_defaults(func=cmd_vis)

    args = parser.parse_args()
    # propagate shared --save-dir into subcommand namespace
    if not hasattr(args, "save_dir"):
        args.save_dir = None
    args.func(args)


def cmd_vis(args):
    # alias so the default func reference works
    cmd_visualise(args)


if __name__ == "__main__":
    main()
