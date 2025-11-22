"""
Multi-Object Tracking System

Implements a multi-object tracker using Kalman filtering for state estimation
and Hungarian algorithm for data association. Tracks are managed with
birth, update, and death lifecycle.
"""

from typing import List, Tuple, Optional
import numpy as np
from .kalman_filter import KalmanFilter3D
from .hungarian import associate_detections_to_tracks


class Track:
    """
    Represents a single tracked object.

    Attributes:
        id: Unique track ID
        kf: Kalman filter for state estimation
        hits: Number of successful updates
        age: Number of frames since track creation
        time_since_update: Frames since last successful update
        state: Current state estimate
        class_id: Object class
    """

    # Global track ID counter
    _next_id = 1

    def __init__(
        self,
        detection: np.ndarray,
        class_id: int = 0,
        dt: float = 0.1,
    ):
        """
        Initialize track from detection.

        Args:
            detection: Initial detection [x, y, z, w, l, h, yaw]
            class_id: Object class ID
            dt: Time step for Kalman filter
        """
        self.id = Track._next_id
        Track._next_id += 1

        self.kf = KalmanFilter3D(detection, dt=dt)
        self.class_id = class_id

        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = detection
        self.history = [detection.copy()]

    def predict(self) -> np.ndarray:
        """
        Predict next state.

        Returns:
            Predicted state
        """
        self.state = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.state

    def update(self, detection: np.ndarray) -> None:
        """
        Update track with new detection.

        Args:
            detection: New detection [x, y, z, w, l, h, yaw]
        """
        self.state = self.kf.update(detection)
        self.hits += 1
        self.time_since_update = 0
        self.history.append(self.state.copy())

    def get_state(self) -> np.ndarray:
        """Get current state estimate."""
        return self.state

    def get_velocity(self) -> np.ndarray:
        """Get velocity estimate."""
        return self.kf.get_velocity()

    def mark_missed(self) -> None:
        """Mark track as missed (no detection matched)."""
        self.time_since_update += 1


class MultiObjectTracker:
    """
    Multi-object tracker using Kalman filter and Hungarian algorithm.

    Manages multiple object tracks with birth, update, and death logic.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        dt: float = 0.1,
    ):
        """
        Initialize multi-object tracker.

        Args:
            max_age: Maximum frames to keep track without updates
            min_hits: Minimum hits before track is confirmed
            iou_threshold: IoU threshold for matching
            dt: Time step in seconds
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt

        self.tracks: List[Track] = []
        self.frame_count = 0

    def update(
        self,
        detections: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Update tracker with new detections.

        Args:
            detections: (N, 7) array of detections [x, y, z, w, l, h, yaw]
            class_ids: (N,) array of class IDs for each detection

        Returns:
            List of confirmed tracks
        """
        self.frame_count += 1

        # Predict new locations for existing tracks
        for track in self.tracks:
            track.predict()

        # Match detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            track_states = np.array([track.get_state() for track in self.tracks])

            matched, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
                detections,
                track_states,
                iou_threshold=self.iou_threshold,
            )
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_tracks = np.arange(len(self.tracks))
            unmatched_dets = np.arange(len(detections))

        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(detections[det_idx])

        # Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            class_id = class_ids[det_idx] if class_ids is not None else 0
            new_track = Track(detections[det_idx], class_id=class_id, dt=self.dt)
            self.tracks.append(new_track)

        # Remove dead tracks
        self.tracks = [
            track for track in self.tracks
            if track.time_since_update < self.max_age
        ]

        # Return confirmed tracks
        confirmed_tracks = [
            track for track in self.tracks
            if track.hits >= self.min_hits or self.frame_count <= self.min_hits
        ]

        return confirmed_tracks

    def get_tracks(self) -> List[Track]:
        """Get all active tracks."""
        return self.tracks

    def get_confirmed_tracks(self) -> List[Track]:
        """Get confirmed tracks only."""
        return [
            track for track in self.tracks
            if track.hits >= self.min_hits
        ]

    def reset(self) -> None:
        """Reset tracker."""
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1


class SimpleTracker:
    """
    Simplified tracker using only IoU matching (no Kalman filter).

    Faster but less robust than full Kalman-based tracker.
    """

    def __init__(
        self,
        max_age: int = 3,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ):
        """
        Initialize simple tracker.

        Args:
            max_age: Maximum frames without update
            min_hits: Minimum hits for confirmation
            iou_threshold: IoU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

        self.tracks = []
        self.next_id = 1

    def update(
        self,
        detections: np.ndarray,
        class_ids: Optional[np.ndarray] = None,
    ) -> List[dict]:
        """
        Update with new detections.

        Args:
            detections: (N, 7) detections
            class_ids: (N,) class IDs

        Returns:
            List of track dictionaries
        """
        # Match detections to tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            track_boxes = np.array([t['box'] for t in self.tracks])

            matched, unmatched_tracks, unmatched_dets = associate_detections_to_tracks(
                detections,
                track_boxes,
                iou_threshold=self.iou_threshold,
            )
        else:
            matched = np.empty((0, 2), dtype=int)
            unmatched_tracks = np.arange(len(self.tracks))
            unmatched_dets = np.arange(len(detections))

        # Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx]['box'] = detections[det_idx]
            self.tracks[track_idx]['hits'] += 1
            self.tracks[track_idx]['time_since_update'] = 0

        # Mark unmatched tracks
        for track_idx in unmatched_tracks:
            self.tracks[track_idx]['time_since_update'] += 1

        # Create new tracks
        for det_idx in unmatched_dets:
            class_id = class_ids[det_idx] if class_ids is not None else 0
            self.tracks.append({
                'id': self.next_id,
                'box': detections[det_idx],
                'class_id': class_id,
                'hits': 1,
                'time_since_update': 0,
            })
            self.next_id += 1

        # Remove dead tracks
        self.tracks = [
            t for t in self.tracks
            if t['time_since_update'] < self.max_age
        ]

        # Return confirmed tracks
        return [
            t for t in self.tracks
            if t['hits'] >= self.min_hits
        ]
