"""
Occupancy Grid for Path Planning

Creates a 2D occupancy grid from 3D object detections for use in path planning.
Occupied cells represent obstacles that should be avoided.
"""

from typing import Tuple, List, Optional
import numpy as np


class OccupancyGrid:
    """
    2D occupancy grid for path planning.

    Represents the environment as a discrete grid where each cell is either
    free (0) or occupied (1). Used for planning collision-free paths.

    Attributes:
        resolution: Cell size in meters
        width: Grid width in meters
        height: Grid height in meters
        origin: Grid origin (x, y) in world coordinates
        grid: 2D occupancy array
    """

    def __init__(
        self,
        resolution: float = 0.2,
        width: float = 100.0,
        height: float = 100.0,
        origin: Tuple[float, float] = (0.0, -50.0),
    ):
        """
        Initialize occupancy grid.

        Args:
            resolution: Cell size in meters
            width: Grid width in meters
            height: Grid height in meters
            origin: Grid origin (x, y) in world frame
        """
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin = origin

        # Calculate grid size
        self.grid_width = int(width / resolution)
        self.grid_height = int(height / resolution)

        # Initialize empty grid (0 = free, 1 = occupied)
        self.grid = np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)

    def update_from_boxes(
        self,
        boxes: np.ndarray,
        safety_margin: float = 2.0,
    ) -> None:
        """
        Update occupancy grid from 3D bounding boxes.

        Projects 3D boxes to 2D and marks occupied cells.

        Args:
            boxes: (N, 7) bounding boxes [x, y, z, w, l, h, yaw]
            safety_margin: Additional margin around boxes in meters
        """
        # Clear previous occupancy
        self.grid.fill(0)

        for box in boxes:
            self._mark_box_occupied(box, safety_margin)

    def _mark_box_occupied(
        self,
        box: np.ndarray,
        safety_margin: float,
    ) -> None:
        """
        Mark grid cells occupied by a bounding box.

        Args:
            box: Bounding box [x, y, z, w, l, h, yaw]
            safety_margin: Safety margin in meters
        """
        x, y, w, l, yaw = box[0], box[1], box[3], box[4], box[6]

        # Add safety margin
        w_safe = w + 2 * safety_margin
        l_safe = l + 2 * safety_margin

        # Get box corners
        corners = self._get_box_corners(x, y, w_safe, l_safe, yaw)

        # Find bounding rectangle in grid coordinates
        grid_corners = [self.world_to_grid(cx, cy) for cx, cy in corners]
        valid_corners = [c for c in grid_corners if c is not None]

        if not valid_corners:
            return

        grid_x = [c[0] for c in valid_corners]
        grid_y = [c[1] for c in valid_corners]

        x_min = max(0, min(grid_x))
        x_max = min(self.grid_width - 1, max(grid_x))
        y_min = max(0, min(grid_y))
        y_max = min(self.grid_height - 1, max(grid_y))

        # Mark rectangle as occupied (simplified - could use actual rotated box)
        self.grid[y_min:y_max+1, x_min:x_max+1] = 1

    def _get_box_corners(
        self,
        x: float,
        y: float,
        w: float,
        l: float,
        yaw: float,
    ) -> List[Tuple[float, float]]:
        """
        Get 2D corners of rotated bounding box.

        Args:
            x, y: Box center
            w, l: Width and length
            yaw: Rotation angle

        Returns:
            List of (x, y) corner coordinates
        """
        # Local corners
        corners_local = [
            (-w/2, -l/2),
            (w/2, -l/2),
            (w/2, l/2),
            (-w/2, l/2),
        ]

        # Rotation matrix
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        # Transform to world frame
        corners_world = []
        for cx, cy in corners_local:
            wx = cos_yaw * cx - sin_yaw * cy + x
            wy = sin_yaw * cx + cos_yaw * cy + y
            corners_world.append((wx, wy))

        return corners_world

    def world_to_grid(
        self,
        x: float,
        y: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Convert world coordinates to grid indices.

        Args:
            x, y: World coordinates

        Returns:
            (grid_x, grid_y) or None if out of bounds
        """
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)

        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return (grid_x, grid_y)
        return None

    def grid_to_world(
        self,
        grid_x: int,
        grid_y: int,
    ) -> Tuple[float, float]:
        """
        Convert grid indices to world coordinates.

        Args:
            grid_x, grid_y: Grid indices

        Returns:
            (x, y) world coordinates
        """
        x = grid_x * self.resolution + self.origin[0] + self.resolution / 2
        y = grid_y * self.resolution + self.origin[1] + self.resolution / 2
        return (x, y)

    def is_occupied(self, grid_x: int, grid_y: int) -> bool:
        """
        Check if grid cell is occupied.

        Args:
            grid_x, grid_y: Grid indices

        Returns:
            True if occupied
        """
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
            return self.grid[grid_y, grid_x] > 0
        return True  # Out of bounds is considered occupied

    def is_free(self, grid_x: int, grid_y: int) -> bool:
        """Check if grid cell is free."""
        return not self.is_occupied(grid_x, grid_y)

    def inflate_obstacles(self, radius: int = 2) -> None:
        """
        Inflate obstacles by given radius.

        Expands occupied regions to create safety buffer.

        Args:
            radius: Inflation radius in grid cells
        """
        from scipy.ndimage import binary_dilation

        # Create structuring element (disk)
        y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
        structure = x**2 + y**2 <= radius**2

        # Dilate occupancy grid
        self.grid = binary_dilation(self.grid, structure=structure).astype(np.uint8)

    def get_grid(self) -> np.ndarray:
        """Get occupancy grid array."""
        return self.grid

    def get_free_space(self) -> np.ndarray:
        """Get binary mask of free space."""
        return 1 - self.grid

    def visualize(self) -> np.ndarray:
        """
        Create visualization of occupancy grid.

        Returns:
            RGB image of grid
        """
        # Create RGB image
        img = np.zeros((self.grid_height, self.grid_width, 3), dtype=np.uint8)

        # Free space = white, occupied = black
        img[self.grid == 0] = [255, 255, 255]
        img[self.grid > 0] = [0, 0, 0]

        return img


def create_occupancy_from_point_cloud(
    points: np.ndarray,
    resolution: float = 0.2,
    width: float = 100.0,
    height: float = 100.0,
    origin: Tuple[float, float] = (0.0, -50.0),
    height_threshold: Tuple[float, float] = (-2.0, 2.0),
) -> OccupancyGrid:
    """
    Create occupancy grid directly from point cloud.

    Alternative to using detected boxes - creates grid from raw points.

    Args:
        points: (N, 3) point cloud
        resolution: Grid resolution
        width, height: Grid dimensions
        origin: Grid origin
        height_threshold: (min, max) z-values to consider

    Returns:
        Occupancy grid
    """
    grid = OccupancyGrid(resolution, width, height, origin)

    # Filter points by height
    z_min, z_max = height_threshold
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    filtered_points = points[mask]

    # Mark occupied cells
    for point in filtered_points:
        cell = grid.world_to_grid(point[0], point[1])
        if cell is not None:
            grid.grid[cell[1], cell[0]] = 1

    return grid
