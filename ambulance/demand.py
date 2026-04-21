import math
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


def make_grid_points(map_bounds, grid_step=1):
    """
    Generate a regular grid of (x, y) sample points over map_bounds.

    map_bounds: (x_min, x_max, y_min, y_max)
    grid_step:  spacing between grid points
    """
    x_min, x_max, y_min, y_max = map_bounds
    xs = np.arange(x_min, x_max + grid_step, grid_step)
    ys = np.arange(y_min, y_max + grid_step, grid_step)
    return [(float(x), float(y)) for x in xs for y in ys]


def make_nearest_coverage(sample_points, ambulance_posts):
    """
    Build J_k: each grid point is covered by its nearest ambulance post.

    Returns: dict mapping hotspot index k -> [nearest post index j]
    """
    return {
        k: [min(range(len(ambulance_posts)),
                key=lambda j, pt=pt: math.dist(pt, ambulance_posts[j]))]
        for k, pt in enumerate(sample_points)
    }


def create_gaussian_field(sample_points, gaussian_params):
    """
    Evaluate a sum of isotropic Gaussian PDFs at each sample point.

    gaussian_params: list of ([mean_x, mean_y], std) tuples
    Returns: np.ndarray of intensities, one per sample point
    """
    pts = np.array(sample_points)
    intensities = np.zeros(len(sample_points))
    for mean, std in gaussian_params:
        rv = multivariate_normal(mean=mean, cov=std ** 2 * np.eye(2))
        intensities += rv.pdf(pts)
    return intensities


def calculate_b_gaussian(ambulance_posts, t, sample_points, map_bounds):
    """
    Return intensity b[k] for each sample point at timestep t.

    Four configurations cycle with period 4:
      0 – 2 Gaussians at opposite corners (deterministic)
      1 – 3 Gaussians at equilateral triangle vertices (deterministic)
      2 – 2 Gaussians with randomised means/stds (seed=2024)
      3 – 3 Gaussians with randomised means/stds (seed=2025)
    """
    x_min, x_max, y_min, y_max = map_bounds
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    r = min(x_max - x_min, y_max - y_min) * 0.35
    std = r / 2.5

    case = t % 4

    if case == 0:
        gaussian_params = [
            ([x_min + r * 0.6, y_min + r * 0.6], std),
            ([x_max - r * 0.6, y_max - r * 0.6], std),
        ]
    elif case == 1:
        gaussian_params = [
            (
                [cx + r * np.cos(np.pi / 2 + 2 * np.pi * k / 3),
                 cy + r * np.sin(np.pi / 2 + 2 * np.pi * k / 3)],
                std,
            )
            for k in range(3)
        ]
    elif case == 2:
        rng = np.random.default_rng(seed=2024)
        means = [
            [x_min + rng.uniform(0, r), y_min + rng.uniform(0, r)],
            [x_max - rng.uniform(0, r), y_max - rng.uniform(0, r)],
        ]
        stds = [rng.uniform(std * 0.5, std * 1.5) for _ in range(2)]
        gaussian_params = list(zip(means, stds))
    else:  # case == 3
        rng = np.random.default_rng(seed=2025)
        means = [
            [cx + rng.uniform(-r, r), cy + rng.uniform(-r, r)]
            for _ in range(3)
        ]
        stds = [rng.uniform(std * 0.5, std * 1.5) for _ in range(3)]
        gaussian_params = list(zip(means, stds))

    return create_gaussian_field(sample_points, gaussian_params) * 100


def calculate_b_centroid(ambulance_posts, t, sample_points, map_bounds, num_ambulances):
    """
    Centroid-adjusted benefit for each grid point at timestep t.

    b'_k = (share of calls at p_k) * (distance saved at p_k)
         = (intensity_k / sum(intensity)) * max(R_{g(k)} - dist(p_k, c_{g(k)}), 0)

    where R_g is the max distance from centroid c_g to any member of cluster g.

    Returns: (b_adjusted, centroids, labels)
    """
    raw_b = calculate_b_gaussian(ambulance_posts, t, sample_points, map_bounds)
    pts = np.array(sample_points)
    weights = np.maximum(raw_b, 0)

    kmeans = KMeans(n_clusters=num_ambulances, random_state=42, n_init=10)
    kmeans.fit(pts, sample_weight=weights)

    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_

    total = weights.sum()
    call_share = weights / total if total > 0 else np.zeros_like(weights)

    distances = np.array([
        math.dist(pts[k], centroids[labels[k]])
        for k in range(len(pts))
    ])
    cluster_radii = np.array([
        distances[labels == g].max() if np.any(labels == g) else 0.0
        for g in range(num_ambulances)
    ])
    distance_saved = np.maximum(cluster_radii[labels] - distances, 0.0)

    b_adjusted = call_share * distance_saved

    return b_adjusted, centroids, labels
