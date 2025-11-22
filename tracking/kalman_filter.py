"""
Kalman Filter for 3D Object Tracking

Implements a Kalman filter for tracking 3D bounding boxes with constant velocity motion model.
The state vector includes position, size, velocity, and rotation.
"""

from typing import Tuple
import numpy as np


class KalmanFilter3D:
    """
    Kalman Filter for tracking 3D bounding boxes.

    State vector: [x, y, z, w, l, h, yaw, vx, vy, vz]
    - (x, y, z): Center position
    - (w, l, h): Dimensions (width, length, height)
    - yaw: Rotation around z-axis
    - (vx, vy, vz): Velocity

    Uses constant velocity motion model for prediction.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        dt: float = 0.1,
        process_noise_pos: float = 2.0,
        process_noise_vel: float = 10.0,
        measurement_noise: float = 1.0,
    ):
        """
        Initialize Kalman filter.

        Args:
            initial_state: Initial state [x, y, z, w, l, h, yaw]
            dt: Time step in seconds
            process_noise_pos: Process noise for position
            process_noise_vel: Process noise for velocity
            measurement_noise: Measurement noise
        """
        self.dt = dt
        self.state_dim = 10  # [x, y, z, w, l, h, yaw, vx, vy, vz]
        self.measurement_dim = 7  # [x, y, z, w, l, h, yaw]

        # Initialize state
        self.x = np.zeros(self.state_dim)
        self.x[:7] = initial_state  # Position, size, orientation
        self.x[7:] = 0  # Initial velocity is zero

        # Initialize covariance matrix
        self.P = np.eye(self.state_dim)
        self.P[:7, :7] *= 10  # High initial uncertainty for position/size
        self.P[7:, 7:] *= 100  # Very high initial uncertainty for velocity

        # State transition matrix (constant velocity model)
        self.F = np.eye(self.state_dim)
        self.F[0, 7] = dt  # x += vx * dt
        self.F[1, 8] = dt  # y += vy * dt
        self.F[2, 9] = dt  # z += vz * dt

        # Measurement matrix (we observe position, size, orientation)
        self.H = np.zeros((self.measurement_dim, self.state_dim))
        self.H[:7, :7] = np.eye(7)

        # Process noise covariance
        self.Q = np.eye(self.state_dim)
        self.Q[:7, :7] *= process_noise_pos
        self.Q[7:, 7:] *= process_noise_vel

        # Measurement noise covariance
        self.R = np.eye(self.measurement_dim) * measurement_noise

    def predict(self) -> np.ndarray:
        """
        Predict next state using motion model.

        Returns:
            Predicted state
        """
        # Predict state
        self.x = self.F @ self.x

        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.x[:7]  # Return observable state

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with new measurement.

        Args:
            measurement: Observed state [x, y, z, w, l, h, yaw]

        Returns:
            Updated state
        """
        # Innovation (measurement residual)
        y = measurement - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

        return self.x[:7]  # Return observable state

    def get_state(self) -> np.ndarray:
        """
        Get current state estimate.

        Returns:
            State [x, y, z, w, l, h, yaw]
        """
        return self.x[:7]

    def get_velocity(self) -> np.ndarray:
        """
        Get current velocity estimate.

        Returns:
            Velocity [vx, vy, vz]
        """
        return self.x[7:]

    def get_covariance(self) -> np.ndarray:
        """
        Get state covariance matrix.

        Returns:
            Covariance matrix
        """
        return self.P[:7, :7]


class ExtendedKalmanFilter3D:
    """
    Extended Kalman Filter for nonlinear motion models.

    Can handle more complex motion patterns than constant velocity.
    Currently implements constant turn rate and velocity (CTRV) model.
    """

    def __init__(
        self,
        initial_state: np.ndarray,
        dt: float = 0.1,
    ):
        """
        Initialize Extended Kalman Filter.

        Args:
            initial_state: Initial state [x, y, z, w, l, h, yaw]
            dt: Time step in seconds
        """
        self.dt = dt
        self.state_dim = 11  # [x, y, z, w, l, h, yaw, v, yaw_rate, vz]
        self.measurement_dim = 7

        # Initialize state
        self.x = np.zeros(self.state_dim)
        self.x[:7] = initial_state
        self.x[7] = 0  # velocity magnitude
        self.x[8] = 0  # yaw rate
        self.x[9] = 0  # vertical velocity

        # Initialize covariance
        self.P = np.eye(self.state_dim) * 10

        # Process and measurement noise
        self.Q = np.eye(self.state_dim) * 2
        self.R = np.eye(self.measurement_dim) * 1

    def predict(self) -> np.ndarray:
        """
        Predict next state using CTRV model.

        Returns:
            Predicted state
        """
        # Extract state variables
        x, y, z = self.x[0], self.x[1], self.x[2]
        yaw = self.x[6]
        v = self.x[7]
        yaw_rate = self.x[8]
        vz = self.x[9]

        # Predict new state
        if abs(yaw_rate) > 0.001:
            # Turning motion
            x_new = x + (v / yaw_rate) * (np.sin(yaw + yaw_rate * self.dt) - np.sin(yaw))
            y_new = y + (v / yaw_rate) * (-np.cos(yaw + yaw_rate * self.dt) + np.cos(yaw))
            yaw_new = yaw + yaw_rate * self.dt
        else:
            # Straight motion
            x_new = x + v * np.cos(yaw) * self.dt
            y_new = y + v * np.sin(yaw) * self.dt
            yaw_new = yaw

        z_new = z + vz * self.dt

        # Update state
        self.x[0] = x_new
        self.x[1] = y_new
        self.x[2] = z_new
        self.x[6] = yaw_new

        # Predict covariance (simplified - use Jacobian for full EKF)
        self.P = self.P + self.Q

        return self.x[:7]

    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update state with measurement.

        Args:
            measurement: Observed state [x, y, z, w, l, h, yaw]

        Returns:
            Updated state
        """
        # Measurement matrix
        H = np.zeros((self.measurement_dim, self.state_dim))
        H[:7, :7] = np.eye(7)

        # Innovation
        y = measurement - H @ self.x

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.x = self.x + K @ y

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

        return self.x[:7]

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.x[:7]

    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity estimate.

        Returns:
            (velocity_magnitude, yaw_rate)
        """
        return self.x[7], self.x[8]
