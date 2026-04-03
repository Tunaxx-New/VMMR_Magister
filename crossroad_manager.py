"""
crossroad_manager.py - Traffic light phase manager with vehicle-type priority weighting.

Priority score per direction is computed as:
    score = sum(vehicle_weight(car) * time_factor(car)) for car in queue

Vehicle weights encode urgency and road impact:
  - Emergency vehicles (ambulance, fire)  : weight 5.0  -> immediate switch
  - Trucks / buses                        : weight 2.5  -> slow to accelerate, need longer green
  - Vans / SUVs                           : weight 1.5
  - Motorcycles                           : weight 0.7  -> fast, low impact
  - Regular cars                          : weight 1.0  (baseline)
  - Unknown / unclassified                : weight 1.0

Time factor grows with wait time so a long-waiting truck eventually
dominates a fresh queue of cars.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from threading import Lock

from vehicle_priority import get_priority
from kalman_priority import KalmanPriorityFilter

logger = logging.getLogger(__name__)

# ── Phase timing ──────────────────────────────────────────────────────────────

MIN_GREEN_SECONDS  = 8.0    # never switch before this
MAX_GREEN_SECONDS  = 45.0   # always switch after this (starvation guard)
YELLOW_SECONDS     = 3.0    # yellow pause before switching
SCORE_SWITCH_RATIO = 1.8    # switch if other axis score >= current * this ratio


def direction_score(cars) -> float:
    """
    Compute the urgency score for a list of TrackedCar objects.

    score = sum( car.priority_weight * time_factor(car.wait_time) )

    priority_weight comes from vehicle_priority.get_priority() set at
    classification time and stored on the TrackedCar dataclass.

    time_factor: 1.0 at 0s, 2.0 at 30s, capped at 3.0 at 60s.
    This ensures a long-waiting truck eventually dominates fresh cars.
    """
    total = 0.0
    for car in cars:
        w = car.priority_weight   # already set from vehicle_priority
        t = min(3.0, 1.0 + car.wait_time / 30.0)
        total += w * t
    return total


# ── Phase state ───────────────────────────────────────────────────────────────

@dataclass
class PhaseState:
    ns:          str   = "green"
    ew:          str   = "red"
    label:       str   = "N-S: Green"
    switched_at: float = field(default_factory=time.time)
    yellow_at:   float = 0.0
    in_yellow:   bool  = False
    ns_score:    float = 0.0
    ew_score:    float = 0.0


# ── Manager ───────────────────────────────────────────────────────────────────

class CrossroadManager:
    """
    Evaluates queue priority scores every second and switches phase when
    the inactive axis score exceeds the active axis score by SCORE_SWITCH_RATIO.

    Priority is driven by vehicle type weights + wait time, so:
      - A bus waiting 10s outweighs 2 cars waiting 3s
      - Emergency vehicles (weight 5x) trigger near-immediate switching
      - Motorcycles (weight 0.7x) are de-prioritised
    """

    def __init__(self, tracker, set_phase_fn, add_log_fn):
        self.tracker       = tracker
        self._set_phase    = set_phase_fn
        self._add_log      = add_log_fn
        self._phase        = PhaseState()
        self._lock         = Lock()
        self._last_tick    = 0.0
        self._tick_interval = 1.0
        self._kf           = KalmanPriorityFilter(
            lookahead_seconds = 8.0,
            process_noise     = 0.5,
            measurement_noise = 1.0,
        )
        self._kf_state: dict = {}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def tick(self) -> None:
        """Rate-limited evaluation — safe to call every frame."""
        now = time.time()
        if now - self._last_tick < self._tick_interval:
            return
        self._last_tick = now
        with self._lock:
            self._evaluate(now)

    def current_phase(self) -> PhaseState:
        with self._lock:
            return self._phase

    def queue_summary(self) -> dict[str, int]:
        return {d: self.tracker.count(d) for d in ("north", "south", "east", "west")}

    def score_summary(self) -> dict[str, float]:
        """Return urgency scores per direction (useful for debugging/display)."""
        return {
            d: round(direction_score(self.tracker.cars_in(d)), 2)
            for d in ("north", "south", "east", "west")
        }

    def kalman_state(self) -> dict:
        """Return last Kalman filter output for both axes."""
        return dict(self._kf_state)

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    def _evaluate(self, now: float) -> None:
        phase   = self._phase
        elapsed = now - phase.switched_at

        # ── Yellow phase: just wait for timer ────────────────────────────
        if phase.in_yellow:
            if now - phase.yellow_at >= YELLOW_SECONDS:
                self._do_switch(now)
            return

        # ── Minimum green guard ──────────────────────────────────────────
        if elapsed < MIN_GREEN_SECONDS:
            return

        # ── Compute raw priority scores for both axes ─────────────────────
        ns_cars = self.tracker.cars_in("north") + self.tracker.cars_in("south")
        ew_cars = self.tracker.cars_in("east")  + self.tracker.cars_in("west")

        ns_raw = direction_score(ns_cars)
        ew_raw = direction_score(ew_cars)

        # ── Kalman filter: smooth + predict future scores ─────────────────
        kf_out = self._kf.update(ns_raw, ew_raw)
        self._kf_state = kf_out

        # Use Kalman-smoothed current score + predicted score blended 60/40
        # This gives advance notice when a queue is building fast
        ns_score = 0.6 * kf_out["ns_smooth"] + 0.4 * kf_out["ns_predicted"]
        ew_score = 0.6 * kf_out["ew_smooth"] + 0.4 * kf_out["ew_predicted"]

        phase.ns_score = round(ns_score, 2)
        phase.ew_score = round(ew_score, 2)

        ns_green      = (phase.ns == "green")
        active_score  = ns_score if ns_green else ew_score
        inactive_score = ew_score if ns_green else ns_score
        inactive_axis = "E-W" if ns_green else "N-S"
        inactive_cars = ew_cars if ns_green else ns_cars

        should_switch = False
        reason        = ""

        # ── Emergency: vehicle weight >= 4 on inactive axis ───────────────
        for car in inactive_cars:
            if car.priority_weight >= 4.0:
                should_switch = True
                reason = (f"Emergency: {car.model_name} on {inactive_axis} "
                          f"(w={car.priority_weight})")
                break

        # ── Max green starvation guard ────────────────────────────────────
        if not should_switch and elapsed >= MAX_GREEN_SECONDS:
            should_switch = True
            reason = f"Max green {elapsed:.0f}s — forcing switch"

        # ── Kalman-smoothed score ratio ───────────────────────────────────
        if not should_switch and inactive_score > 0:
            ratio = inactive_score / max(active_score, 0.1)
            if ratio >= SCORE_SWITCH_RATIO:
                should_switch = True
                dominant = _top_vehicle(inactive_cars)
                vel_info = (f"  vel={kf_out['ew_velocity' if ns_green else 'ns_velocity']:+.2f}/s")
                reason = (
                    f"{inactive_axis} Kalman score={inactive_score:.1f} "
                    f"vs active={active_score:.1f} "
                    f"(ratio={ratio:.1f}x){vel_info}"
                    + (f" — {dominant}" if dominant else "")
                )

        if should_switch:
            logger.info("Phase switch: %s", reason)
            self._add_log("system", f"Switch: {reason}", "warn")
            self._start_yellow(now)

    def _start_yellow(self, now: float) -> None:
        phase = self._phase
        if phase.ns == "green":
            phase.ns = "yellow"
        else:
            phase.ew = "yellow"
        phase.label     = "Switching..."
        phase.in_yellow = True
        phase.yellow_at = now
        self._set_phase(phase.ns, phase.ew, phase.label)

    def _do_switch(self, now: float) -> None:
        phase = self._phase
        if phase.ns in ("yellow", "red"):
            phase.ns    = "green"
            phase.ew    = "red"
            phase.label = "N-S: Green"
        else:
            phase.ns    = "red"
            phase.ew    = "green"
            phase.label = "E-W: Green"
        phase.in_yellow   = False
        phase.switched_at = now
        self._set_phase(phase.ns, phase.ew, phase.label)
        self._add_log("system", f"Phase: {phase.label}", "info")
        logger.info("Phase: %s", phase.label)


# ── Helper ────────────────────────────────────────────────────────────────────

def _top_vehicle(cars) -> str:
    """Return the model name of the highest-weight car in the list."""
    if not cars:
        return ""
    return max(cars, key=lambda c: c.priority_weight).model_name