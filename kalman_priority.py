"""
kalman_priority.py - Kalman filter for traffic light phase priority estimation.

Models the queue urgency of each direction as a dynamical system:

    State vector x = [score, score_velocity]
        score          : current weighted urgency score
        score_velocity : rate of score change (cars arriving / leaving)

    Process model (constant velocity):
        score(t+1)          = score(t) + dt * velocity(t)
        velocity(t+1)       = velocity(t)  + process_noise

    Measurement:
        z(t) = score_observed = direction_score(tracker.cars_in(direction))

The Kalman filter smooths noisy per-second score jumps and predicts
future urgency, giving the crossroad manager advance notice of when
a direction is about to become critical — before it actually spills over.

Usage:
    kf = KalmanPriorityFilter()
    predicted_ns, predicted_ew = kf.update(ns_score, ew_score, dt)
    # Use predicted scores instead of raw scores for phase decisions
"""

from __future__ import annotations

import logging
import time
import numpy as np

logger = logging.getLogger(__name__)


class DirectionKalman:
    """
    1D Kalman filter tracking [score, score_velocity] for one direction axis.

    State:   x = [s, v]  (score, velocity)
    Measure: z = [s]      (observed score)
    """

    def __init__(
        self,
        process_noise_score: float = 0.5,
        process_noise_vel:   float = 0.2,
        measurement_noise:   float = 1.0,
    ):
        # State vector
        self.x = np.array([0.0, 0.0])        # [score, velocity]

        # State covariance
        self.P = np.eye(2) * 5.0

        # Measurement matrix: we only observe score
        self.H = np.array([[1.0, 0.0]])

        # Measurement noise covariance
        self.R = np.array([[measurement_noise]])

        # Process noise covariance
        self.Q = np.array([
            [process_noise_score, 0.0],
            [0.0,                 process_noise_vel],
        ])

        self._last_t = time.time()

    def predict(self, dt: float) -> float:
        """Advance state by dt seconds. Returns predicted score."""
        # State transition matrix (constant velocity model)
        F = np.array([
            [1.0, dt],
            [0.0, 1.0],
        ])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        return float(self.x[0])

    def update(self, observed_score: float) -> float:
        """
        Incorporate a new measurement. Returns corrected score estimate.
        """
        z = np.array([[observed_score]])

        # Innovation
        y  = z - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        K  = self.P @ self.H.T @ np.linalg.inv(S)

        # Correct state
        self.x = self.x + (K @ y).flatten()
        self.P = (np.eye(2) - K @ self.H) @ self.P

        # Clamp score to non-negative
        self.x[0] = max(0.0, self.x[0])

        return float(self.x[0])

    @property
    def score(self) -> float:
        return float(self.x[0])

    @property
    def velocity(self) -> float:
        """Rate of score change (positive = queue building, negative = clearing)."""
        return float(self.x[1])

    def predict_in(self, seconds: float) -> float:
        """Predict what the score will be N seconds from now."""
        return max(0.0, self.score + self.velocity * seconds)


class KalmanPriorityFilter:
    """
    Maintains Kalman filters for both N-S and E-W axes and provides
    smoothed + predicted priority scores to the crossroad manager.

    Parameters
    ----------
    lookahead_seconds : float
        How far ahead to predict urgency (used for proactive switching).
    process_noise : float
        How much the model trusts state changes (higher = tracks faster).
    measurement_noise : float
        How much the model trusts observations (lower = smoother output).
    """

    def __init__(
        self,
        lookahead_seconds: float = 8.0,
        process_noise:     float = 0.5,
        measurement_noise: float = 1.0,
    ):
        self.lookahead = lookahead_seconds
        self._ns = DirectionKalman(process_noise, process_noise * 0.4, measurement_noise)
        self._ew = DirectionKalman(process_noise, process_noise * 0.4, measurement_noise)
        self._last_tick = time.time()

    def update(
        self,
        ns_observed: float,
        ew_observed: float,
    ) -> dict[str, float]:
        """
        Feed observed scores, return dict with smoothed + predicted values.

        Returns
        -------
        {
            "ns_smooth"   : Kalman-corrected N-S score,
            "ew_smooth"   : Kalman-corrected E-W score,
            "ns_predicted": predicted N-S score in lookahead_seconds,
            "ew_predicted": predicted E-W score in lookahead_seconds,
            "ns_velocity" : rate of N-S score change,
            "ew_velocity" : rate of E-W score change,
            "dt"          : seconds since last update,
        }
        """
        now = time.time()
        dt  = now - self._last_tick
        self._last_tick = now

        # Predict step
        self._ns.predict(dt)
        self._ew.predict(dt)

        # Update step
        ns_smooth = self._ns.update(ns_observed)
        ew_smooth = self._ew.update(ew_observed)

        return {
            "ns_smooth":    round(ns_smooth, 3),
            "ew_smooth":    round(ew_smooth, 3),
            "ns_predicted": round(self._ns.predict_in(self.lookahead), 3),
            "ew_predicted": round(self._ew.predict_in(self.lookahead), 3),
            "ns_velocity":  round(self._ns.velocity, 3),
            "ew_velocity":  round(self._ew.velocity, 3),
            "dt":           round(dt, 3),
        }

    def reset(self) -> None:
        self._ns = DirectionKalman()
        self._ew = DirectionKalman()