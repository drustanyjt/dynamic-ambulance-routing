import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds


def euclidean_distance(a, b):
    return math.dist(a, b)


def calculate_c(ambulance_assignments, ambulance_posts, t):
    """
    Cost matrix c[i][j] = distance from ambulance i's current post to post j.

    ambulance_assignments: list of current post indices, one per ambulance
    ambulance_posts: list of (x, y) post locations
    t: current timestep (unused here, reserved for future extensions)
    """
    num_ambulances = len(ambulance_assignments)
    num_posts = len(ambulance_posts)
    c = []
    for i in range(num_ambulances):
        current_loc = ambulance_posts[ambulance_assignments[i]]
        c.append([math.dist(current_loc, ambulance_posts[j]) for j in range(num_posts)])
    return c


def solve_ambulance_milp(num_ambulances, ambulance_posts, c, b, J_k):
    """
    Solve the ambulance post assignment MILP.

    Variables (all binary):
      x[i][j]  – ambulance i assigned to post j   (index: i*num_posts + j)
      y[k]     – hotspot k is covered              (index: n_x + k)

    Objective:
      min  sum_ij c[i][j]*x[i][j]  -  sum_k b[k]*y[k]

    Constraints:
      sum_j x[i][j] = 1            for all i   (each ambulance assigned once)
      sum_i x[i][j] <= 1           for all j   (at most one ambulance per post)
      sum_{i, j in J_k[k]} x[i][j] >= y[k]    (coverage linkage)

    Returns:
      list of (ambulance_index, post_index) pairs
    """
    num_posts = len(ambulance_posts)
    num_hotspots = len(b)
    n_x = num_ambulances * num_posts
    n_vars = n_x + num_hotspots

    # Objective
    c_obj = np.zeros(n_vars)
    for i in range(num_ambulances):
        for j in range(num_posts):
            c_obj[i * num_posts + j] = c[i][j]
    for k in range(num_hotspots):
        c_obj[n_x + k] = -b[k]

    # Constraints
    rows, lbs, ubs = [], [], []

    # Each ambulance assigned to exactly one post
    for i in range(num_ambulances):
        row = np.zeros(n_vars)
        for j in range(num_posts):
            row[i * num_posts + j] = 1.0
        rows.append(row)
        lbs.append(1.0)
        ubs.append(1.0)

    # At most one ambulance per post
    for j in range(num_posts):
        row = np.zeros(n_vars)
        for i in range(num_ambulances):
            row[i * num_posts + j] = 1.0
        rows.append(row)
        lbs.append(0.0)
        ubs.append(1.0)

    # Coverage linkage: sum_{i, j in J_k[k]} x[i][j] - y[k] >= 0
    for k in range(num_hotspots):
        covering = J_k.get(k, [])
        row = np.zeros(n_vars)
        for i in range(num_ambulances):
            for j in covering:
                row[i * num_posts + j] = 1.0
        row[n_x + k] = -1.0
        rows.append(row)
        lbs.append(0.0)
        ubs.append(np.inf)

    constraints = LinearConstraint(np.array(rows), lbs, ubs)
    bounds = Bounds(lb=0, ub=1)
    integrality = np.ones(n_vars)

    result = milp(c_obj, constraints=constraints, integrality=integrality, bounds=bounds)
    if not result.success:
        raise RuntimeError(f"MILP solver failed: {result.message}")

    x_vals = result.x[:n_x].reshape(num_ambulances, num_posts)
    return [(i, j)
            for i in range(num_ambulances)
            for j in range(num_posts)
            if x_vals[i, j] > 0.5]
