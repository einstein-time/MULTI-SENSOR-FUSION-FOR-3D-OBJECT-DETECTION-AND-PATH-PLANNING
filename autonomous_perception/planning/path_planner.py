"""
Path Planning Algorithms

Implements A* and Hybrid A* for finding collision-free paths in occupancy grids.
"""

from typing import List, Tuple, Optional, Set
import numpy as np
import heapq
from .occupancy_grid import OccupancyGrid


class Node:
    """Node for A* search."""

    def __init__(
        self,
        x: int,
        y: int,
        cost: float = 0.0,
        heuristic: float = 0.0,
        parent: Optional['Node'] = None,
    ):
        """
        Initialize node.

        Args:
            x, y: Grid coordinates
            cost: Cost from start (g-score)
            heuristic: Heuristic cost to goal (h-score)
            parent: Parent node
        """
        self.x = x
        self.y = y
        self.cost = cost
        self.heuristic = heuristic
        self.parent = parent

    @property
    def f_score(self) -> float:
        """Total cost (f = g + h)."""
        return self.cost + self.heuristic

    def __lt__(self, other: 'Node') -> bool:
        """Compare nodes by f-score."""
        return self.f_score < other.f_score

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Node):
            return False
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        """Hash for set/dict."""
        return hash((self.x, self.y))


class AStarPlanner:
    """
    A* path planner for grid-based navigation.

    Finds optimal path from start to goal while avoiding obstacles.
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        allow_diagonal: bool = True,
    ):
        """
        Initialize A* planner.

        Args:
            occupancy_grid: Occupancy grid
            allow_diagonal: Allow diagonal movements
        """
        self.grid = occupancy_grid
        self.allow_diagonal = allow_diagonal

        # Movement directions (4 or 8 connected)
        if allow_diagonal:
            self.motions = [
                (0, 1, 1.0),    # Up
                (0, -1, 1.0),   # Down
                (1, 0, 1.0),    # Right
                (-1, 0, 1.0),   # Left
                (1, 1, 1.414),  # Diagonal
                (1, -1, 1.414),
                (-1, 1, 1.414),
                (-1, -1, 1.414),
            ]
        else:
            self.motions = [
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 0, 1.0),
                (-1, 0, 1.0),
            ]

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path from start to goal.

        Args:
            start: Start position (x, y) in world coordinates
            goal: Goal position (x, y) in world coordinates

        Returns:
            Path as list of (x, y) waypoints or None if no path exists
        """
        # Convert to grid coordinates
        start_grid = self.grid.world_to_grid(*start)
        goal_grid = self.grid.world_to_grid(*goal)

        if start_grid is None or goal_grid is None:
            return None

        # Check if start/goal are free
        if self.grid.is_occupied(*start_grid) or self.grid.is_occupied(*goal_grid):
            return None

        # Run A* search
        path_grid = self._search(start_grid, goal_grid)

        if path_grid is None:
            return None

        # Convert back to world coordinates
        path_world = [self.grid.grid_to_world(x, y) for x, y in path_grid]

        return path_world

    def _search(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        A* search algorithm.

        Args:
            start: Start grid position
            goal: Goal grid position

        Returns:
            Path as list of grid coordinates
        """
        # Initialize
        start_node = Node(
            start[0], start[1],
            cost=0.0,
            heuristic=self._heuristic(start, goal),
        )

        open_set = [start_node]
        closed_set: Set[Tuple[int, int]] = set()
        g_scores = {(start[0], start[1]): 0.0}

        while open_set:
            # Get node with lowest f-score
            current = heapq.heappop(open_set)
            current_pos = (current.x, current.y)

            # Check if goal reached
            if current_pos == goal:
                return self._reconstruct_path(current)

            # Add to closed set
            closed_set.add(current_pos)

            # Explore neighbors
            for dx, dy, cost in self.motions:
                next_x = current.x + dx
                next_y = current.y + dy
                next_pos = (next_x, next_y)

                # Check bounds
                if not (0 <= next_x < self.grid.grid_width and
                        0 <= next_y < self.grid.grid_height):
                    continue

                # Check if already visited
                if next_pos in closed_set:
                    continue

                # Check if occupied
                if self.grid.is_occupied(next_x, next_y):
                    continue

                # Calculate new cost
                new_cost = current.cost + cost

                # Check if this path is better
                if next_pos in g_scores and new_cost >= g_scores[next_pos]:
                    continue

                # Create neighbor node
                neighbor = Node(
                    next_x, next_y,
                    cost=new_cost,
                    heuristic=self._heuristic(next_pos, goal),
                    parent=current,
                )

                g_scores[next_pos] = new_cost
                heapq.heappush(open_set, neighbor)

        # No path found
        return None

    def _heuristic(
        self,
        pos: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> float:
        """
        Heuristic function (Euclidean distance).

        Args:
            pos: Current position
            goal: Goal position

        Returns:
            Heuristic cost
        """
        dx = pos[0] - goal[0]
        dy = pos[1] - goal[1]
        return np.sqrt(dx**2 + dy**2)

    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from goal node.

        Args:
            node: Goal node

        Returns:
            Path from start to goal
        """
        path = []
        current = node

        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        return list(reversed(path))


class HybridAStarPlanner:
    """
    Hybrid A* planner with continuous state space.

    Unlike regular A*, this considers vehicle kinematics and orientation.
    More suitable for actual autonomous driving.
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        turning_radius: float = 5.0,
        step_size: float = 1.0,
    ):
        """
        Initialize Hybrid A* planner.

        Args:
            occupancy_grid: Occupancy grid
            turning_radius: Minimum turning radius in meters
            step_size: Step size for motion primitives
        """
        self.grid = occupancy_grid
        self.turning_radius = turning_radius
        self.step_size = step_size

        # Discretize heading (16 directions)
        self.num_headings = 16
        self.heading_resolution = 2 * np.pi / self.num_headings

    def plan(
        self,
        start: Tuple[float, float, float],
        goal: Tuple[float, float, float],
    ) -> Optional[List[Tuple[float, float, float]]]:
        """
        Plan path with continuous state space.

        Args:
            start: Start (x, y, heading)
            goal: Goal (x, y, heading)

        Returns:
            Path as list of (x, y, heading) waypoints
        """
        # Simplified implementation - full hybrid A* is complex
        # For production, use libraries like OMPL

        # Fall back to regular A* and add heading interpolation
        path_2d = AStarPlanner(self.grid, allow_diagonal=True).plan(
            start[:2], goal[:2]
        )

        if path_2d is None:
            return None

        # Add heading to path (simple interpolation)
        path_3d = []
        for i, (x, y) in enumerate(path_2d):
            if i < len(path_2d) - 1:
                dx = path_2d[i+1][0] - x
                dy = path_2d[i+1][1] - y
                heading = np.arctan2(dy, dx)
            else:
                heading = goal[2]

            path_3d.append((x, y, heading))

        return path_3d


class RRTPlanner:
    """
    Rapidly-exploring Random Tree (RRT) planner.

    Alternative to A* for high-dimensional spaces or when A* is too slow.
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        max_iterations: int = 1000,
        step_size: float = 2.0,
        goal_sample_rate: float = 0.1,
    ):
        """
        Initialize RRT planner.

        Args:
            occupancy_grid: Occupancy grid
            max_iterations: Maximum iterations
            step_size: Step size for tree expansion
            goal_sample_rate: Probability of sampling goal
        """
        self.grid = occupancy_grid
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate

    def plan(
        self,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """
        Plan path using RRT.

        Args:
            start: Start position
            goal: Goal position

        Returns:
            Path or None
        """
        # Build RRT tree
        tree = {start: None}

        for _ in range(self.max_iterations):
            # Sample random point
            if np.random.random() < self.goal_sample_rate:
                sample = goal
            else:
                sample = self._random_sample()

            # Find nearest node in tree
            nearest = self._nearest_node(tree, sample)

            # Steer towards sample
            new_node = self._steer(nearest, sample)

            # Check collision
            if self._is_collision_free(nearest, new_node):
                tree[new_node] = nearest

                # Check if goal reached
                dist_to_goal = np.linalg.norm(
                    np.array(new_node) - np.array(goal)
                )
                if dist_to_goal < self.step_size:
                    tree[goal] = new_node
                    return self._extract_path(tree, start, goal)

        return None

    def _random_sample(self) -> Tuple[float, float]:
        """Sample random point in grid."""
        x = np.random.uniform(
            self.grid.origin[0],
            self.grid.origin[0] + self.grid.width
        )
        y = np.random.uniform(
            self.grid.origin[1],
            self.grid.origin[1] + self.grid.height
        )
        return (x, y)

    def _nearest_node(
        self,
        tree: dict,
        point: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Find nearest node in tree."""
        nodes = list(tree.keys())
        distances = [
            np.linalg.norm(np.array(node) - np.array(point))
            for node in nodes
        ]
        return nodes[np.argmin(distances)]

    def _steer(
        self,
        from_node: Tuple[float, float],
        to_node: Tuple[float, float],
    ) -> Tuple[float, float]:
        """Steer from one node towards another."""
        direction = np.array(to_node) - np.array(from_node)
        distance = np.linalg.norm(direction)

        if distance < self.step_size:
            return to_node

        direction = direction / distance * self.step_size
        new_node = np.array(from_node) + direction

        return tuple(new_node)

    def _is_collision_free(
        self,
        from_node: Tuple[float, float],
        to_node: Tuple[float, float],
    ) -> bool:
        """Check if path between nodes is collision-free."""
        # Sample points along line
        num_samples = int(
            np.linalg.norm(np.array(to_node) - np.array(from_node)) /
            self.grid.resolution
        )

        for i in range(num_samples + 1):
            t = i / max(num_samples, 1)
            point = (
                from_node[0] + t * (to_node[0] - from_node[0]),
                from_node[1] + t * (to_node[1] - from_node[1]),
            )

            cell = self.grid.world_to_grid(*point)
            if cell is None or self.grid.is_occupied(*cell):
                return False

        return True

    def _extract_path(
        self,
        tree: dict,
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        """Extract path from tree."""
        path = [goal]
        current = goal

        while current != start:
            current = tree[current]
            if current is None:
                break
            path.append(current)

        return list(reversed(path))
