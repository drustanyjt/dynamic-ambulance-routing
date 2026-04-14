import math
import numpy as np

from .model import calculate_c, solve_ambulance_milp


def polygon_vertices(n, center, radius):
    """Return n evenly-spaced vertices of a regular polygon on a circle."""
    angles = np.linspace(0, 2 * np.pi, n + 1)[:-1]
    return [
        (center[0] + radius * np.cos(a), center[1] + radius * np.sin(a))
        for a in angles
    ]


def run_simulation(
    num_timesteps,
    num_ambulances,
    ambulance_posts,
    call_hotspots,
    J_k,
    calculate_b_fn,
    calculate_c_fn=None,
    verbose=True,
):
    """
    Receding horizon simulation.

    At each timestep:
      1. Compute cost matrix from current ambulance positions.
      2. Compute benefit vector via calculate_b_fn.
      3. Solve MILP to get new assignments.
      4. Move ambulances to assigned posts.

    Parameters
    ----------
    num_timesteps   : int
    num_ambulances  : int
    ambulance_posts : list of (x, y)
    call_hotspots   : list of (x, y)   (used for visualisation only)
    J_k             : dict  k -> [post indices that cover hotspot k]
    calculate_b_fn  : (ambulance_assignments, ambulance_posts, t) -> array-like
    calculate_c_fn  : (ambulance_assignments, ambulance_posts, t) -> 2-D list
                      defaults to Euclidean distance
    verbose         : print assignment table to stdout

    Returns
    -------
    list of length num_timesteps, each element a list of (ambulance_idx, post_idx)
    """
    if calculate_c_fn is None:
        calculate_c_fn = calculate_c

    assignments_history = []
    ambulance_assignments = [0] * num_ambulances  # start all at post 0

    for t in range(num_timesteps):
        c = (
            [[0.0] * len(ambulance_posts) for _ in range(num_ambulances)]
            if t == 0
            else calculate_c_fn(ambulance_assignments, ambulance_posts, t)
        )
        b = calculate_b_fn(ambulance_assignments, ambulance_posts, t)

        assignments = solve_ambulance_milp(num_ambulances, ambulance_posts, c, b, J_k)
        ambulance_assignments = [j for _, j in assignments]
        assignments_history.append(assignments)

        if verbose:
            posts = [j for _, j in sorted(assignments)]
            print(f"t={t}: ambulances -> posts {posts}")

    return assignments_history
