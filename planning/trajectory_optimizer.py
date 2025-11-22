"""
Trajectory Optimization and Smoothing

Converts discrete waypoints into smooth, kinematically feasible trajectories.
Considers velocity, acceleration, and jerk constraints.
"""

from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import CubicSpline, splprep, splev
from scipy.optimize import minimize


class TrajectoryOptimizer:
    """
    Optimize trajectory for smoothness and kinematic feasibility.

    Converts waypoint path into smooth trajectory with velocity profile.
    """

    def __init__(
        self,
        max_velocity: float = 15.0,
        max_acceleration: float = 3.0,
        max_jerk: float = 2.0,
        dt: float = 0.1,
    ):
        """
        Initialize trajectory optimizer.

        Args:
            max_velocity: Maximum velocity in m/s
            max_acceleration: Maximum acceleration in m/s^2
            max_jerk: Maximum jerk in m/s^3
            dt: Time step for trajectory
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk
        self.dt = dt

    def optimize(
        self,
        waypoints: List[Tuple[float, float]],
        initial_velocity: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Optimize trajectory from waypoints.

        Args:
            waypoints: List of (x, y) waypoints
            initial_velocity: Initial velocity

        Returns:
            positions: (N, 2) trajectory positions
            velocities: (N,) velocity profile
            timestamps: (N,) time stamps
        """
        if len(waypoints) < 2:
            raise ValueError("Need at least 2 waypoints")

        waypoints = np.array(waypoints)

        # Smooth path using spline
        positions = self._smooth_path(waypoints)

        # Compute velocity profile
        velocities = self._compute_velocity_profile(
            positions,
            initial_velocity,
        )

        # Compute timestamps
        timestamps = np.arange(len(positions)) * self.dt

        return positions, velocities, timestamps

    def _smooth_path(
        self,
        waypoints: np.ndarray,
        num_points: Optional[int] = None,
    ) -> np.ndarray:
        """
        Smooth path using cubic spline.

        Args:
            waypoints: (N, 2) waypoints
            num_points: Number of points in smoothed path

        Returns:
            Smoothed path
        """
        if len(waypoints) < 4:
            # Not enough points for spline, use linear interpolation
            if num_points is None:
                return waypoints

            distances = np.cumsum([0] + [
                np.linalg.norm(waypoints[i+1] - waypoints[i])
                for i in range(len(waypoints) - 1)
            ])
            total_distance = distances[-1]

            if total_distance == 0:
                return waypoints

            sample_distances = np.linspace(0, total_distance, num_points)
            smoothed = np.array([
                np.interp(sample_distances, distances, waypoints[:, 0]),
                np.interp(sample_distances, distances, waypoints[:, 1]),
            ]).T

            return smoothed

        # Use B-spline for smoothing
        try:
            # Parameterize by arc length
            tck, u = splprep([waypoints[:, 0], waypoints[:, 1]], s=0, k=3)

            if num_points is None:
                # Estimate based on path length
                path_length = np.sum([
                    np.linalg.norm(waypoints[i+1] - waypoints[i])
                    for i in range(len(waypoints) - 1)
                ])
                num_points = max(int(path_length / 0.5), 100)

            # Evaluate spline
            u_new = np.linspace(0, 1, num_points)
            smoothed = np.array(splev(u_new, tck)).T

            return smoothed

        except Exception:
            # Fall back to linear interpolation
            return waypoints

    def _compute_velocity_profile(
        self,
        positions: np.ndarray,
        initial_velocity: float,
    ) -> np.ndarray:
        """
        Compute velocity profile respecting kinematic constraints.

        Uses trapezoidal velocity profile with acceleration limits.

        Args:
            positions: (N, 2) trajectory positions
            initial_velocity: Initial velocity

        Returns:
            Velocity profile (N,)
        """
        num_points = len(positions)
        velocities = np.zeros(num_points)
        velocities[0] = initial_velocity

        # Compute path segments
        segment_lengths = np.linalg.norm(
            positions[1:] - positions[:-1],
            axis=1
        )

        # Forward pass: accelerate
        for i in range(1, num_points):
            # Maximum velocity achievable with acceleration limit
            v_max = np.sqrt(
                velocities[i-1]**2 +
                2 * self.max_acceleration * segment_lengths[i-1]
            )

            velocities[i] = min(v_max, self.max_velocity)

        # Backward pass: decelerate for upcoming constraints
        velocities[-1] = 0  # Stop at end

        for i in range(num_points - 2, -1, -1):
            # Maximum velocity that can decelerate in time
            v_max = np.sqrt(
                velocities[i+1]**2 +
                2 * self.max_acceleration * segment_lengths[i]
            )

            velocities[i] = min(velocities[i], v_max)

        return velocities


def compute_trajectory(
    path: List[Tuple[float, float]],
    horizon: float = 5.0,
    dt: float = 0.1,
    max_velocity: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute trajectory from path.

    Simple utility function for quick trajectory generation.

    Args:
        path: List of waypoints
        horizon: Time horizon in seconds
        dt: Time step
        max_velocity: Maximum velocity

    Returns:
        positions: Trajectory positions
        timestamps: Time stamps
    """
    if len(path) < 2:
        path_array = np.array(path if path else [[0, 0]])
        timestamps = np.array([0])
        return path_array, timestamps

    optimizer = TrajectoryOptimizer(
        max_velocity=max_velocity,
        dt=dt,
    )

    positions, velocities, timestamps = optimizer.optimize(path)

    # Limit to horizon
    max_idx = int(horizon / dt)
    positions = positions[:max_idx]
    timestamps = timestamps[:max_idx]

    return positions, timestamps


def smooth_trajectory(
    positions: np.ndarray,
    window_size: int = 5,
) -> np.ndarray:
    """
    Smooth trajectory using moving average.

    Args:
        positions: (N, 2) trajectory
        window_size: Smoothing window size

    Returns:
        Smoothed trajectory
    """
    if len(positions) < window_size:
        return positions

    smoothed = np.copy(positions)

    for i in range(len(positions)):
        start = max(0, i - window_size // 2)
        end = min(len(positions), i + window_size // 2 + 1)
        smoothed[i] = positions[start:end].mean(axis=0)

    return smoothed


def check_collision(
    trajectory: np.ndarray,
    obstacles: np.ndarray,
    safety_margin: float = 2.0,
) -> bool:
    """
    Check if trajectory collides with obstacles.

    Args:
        trajectory: (N, 2) trajectory positions
        obstacles: (M, 7) obstacle bounding boxes
        safety_margin: Safety margin in meters

    Returns:
        True if collision detected
    """
    for pos in trajectory:
        for obs in obstacles:
            # Simple distance check
            distance = np.linalg.norm(pos - obs[:2])
            min_distance = np.sqrt(obs[3]**2 + obs[4]**2) / 2 + safety_margin

            if distance < min_distance:
                return True

    return False


def compute_curvature(
    positions: np.ndarray,
) -> np.ndarray:
    """
    Compute path curvature at each point.

    High curvature indicates sharp turns that may require lower speed.

    Args:
        positions: (N, 2) path positions

    Returns:
        Curvature values (N,)
    """
    if len(positions) < 3:
        return np.zeros(len(positions))

    # Compute first and second derivatives
    dx = np.gradient(positions[:, 0])
    dy = np.gradient(positions[:, 1])
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
    numerator = np.abs(dx * ddy - dy * ddx)
    denominator = (dx**2 + dy**2)**(3/2)

    curvature = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0
    )

    return curvature


def compute_velocity_from_curvature(
    curvature: np.ndarray,
    max_velocity: float = 15.0,
    max_lateral_accel: float = 2.0,
) -> np.ndarray:
    """
    Compute safe velocity based on path curvature.

    Lower velocity for sharp turns to maintain lateral acceleration limits.

    Args:
        curvature: Path curvature
        max_velocity: Maximum velocity
        max_lateral_accel: Maximum lateral acceleration

    Returns:
        Velocity profile
    """
    # v = sqrt(a_lat / kappa)
    # where kappa is curvature
    velocities = np.sqrt(
        np.divide(
            max_lateral_accel,
            curvature,
            out=np.full_like(curvature, max_velocity),
            where=curvature > 1e-6
        )
    )

    velocities = np.clip(velocities, 0, max_velocity)

    return velocities
