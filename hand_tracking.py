"""Hand landmark tracking utilities built on MediaPipe with smoothing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import time

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandInfo:
    index_tip: tuple[int, int]
    pinch_strength: float
    velocity: tuple[float, float]
    handedness: str


class HandTracker:
    def __init__(self, max_hands: int = 1, cutoff_hz: float = 10.0) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._handedness = mp.solutions.hands
        self._smoothed_tip: Optional[np.ndarray] = None
        self._last_tip: Optional[np.ndarray] = None
        self._pinch_lp: float = 0.0
        self._last_time: Optional[float] = None
        self._cutoff = cutoff_hz

    def _reset(self) -> None:
        self._smoothed_tip = None
        self._last_tip = None
        self._pinch_lp = 0.0
        self._last_time = None

    def detect(self, frame: cv2.Mat) -> Optional[HandInfo]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)
        if not results.multi_hand_landmarks:
            self._reset()
            return None
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = (
            results.multi_handedness[0].classification[0].label
            if results.multi_handedness
            else "unknown"
        )
        h, w, _ = frame.shape
        index_tip = hand_landmarks.landmark[self._handedness.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[self._handedness.HandLandmark.THUMB_TIP]
        current_tip = np.array([index_tip.x * w, index_tip.y * h], dtype=np.float32)
        tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
        pinch_dist = ((current_tip[0] - tx) ** 2 + (current_tip[1] - ty) ** 2) ** 0.5
        pinch_strength = float(max(0.0, min(1.0, 1 - pinch_dist / 160)))

        now = time.perf_counter()
        if self._last_time is None:
            dt = 1 / 30
        else:
            dt = max(1e-3, now - self._last_time)
        self._last_time = now

        if self._smoothed_tip is None:
            self._smoothed_tip = current_tip.copy()
        else:
            alpha = 1 - np.exp(-dt * self._cutoff)
            self._smoothed_tip = (1 - alpha) * self._smoothed_tip + alpha * current_tip

        if self._last_tip is None:
            velocity = np.zeros(2, dtype=np.float32)
        else:
            velocity = (self._smoothed_tip - self._last_tip) / dt
        self._last_tip = self._smoothed_tip.copy()

        self._pinch_lp += (pinch_strength - self._pinch_lp) * 0.35

        return HandInfo(
            index_tip=(int(self._smoothed_tip[0]), int(self._smoothed_tip[1])),
            pinch_strength=float(self._pinch_lp),
            velocity=(float(velocity[0]), float(velocity[1])),
            handedness=handedness,
        )
