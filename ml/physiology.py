"""
physiology.py
-------------
Deterministic model of arterial CO₂ (PaCO₂) accumulation during apnea.

Physiology reference
~~~~~~~~~~~~~~~~~~~~
  • During the first 60 s of apnea PaCO₂ rises by ~12 mmHg.
  • Thereafter it rises at ~3.4 mmHg per additional minute.

Equation
~~~~~~~~
  PaCO₂(t) = PaCO₂(0) + ΔPaCO₂(t)

  ΔPaCO₂(t) =
      12 × (t / 60)                          if  0 < t ≤ 60 s
      12 + 3.4 × ((t − 60) / 60)            if  t > 60 s

Alert trigger (fusion module guardrail)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  trigger_alert = (ml_detection == True) AND (ΔPaCO₂ > ALERT_THRESHOLD)

  Default ALERT_THRESHOLD = 4.0 mmHg  (≈ 20 s into apnea, well within 1st min)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INITIAL_PACO2: float = 40.0   # baseline arterial PaCO₂ (mmHg)
FIRST_MIN_RISE: float = 12.0  # mmHg rise during first 60 s of apnea
SUBSEQUENT_RISE: float = 3.4  # mmHg per additional minute after 60 s
ALERT_THRESHOLD: float = 4.0  # mmHg rise that triggers a respiratory alert


# ---------------------------------------------------------------------------
# Stateless helpers
# ---------------------------------------------------------------------------

def estimate_paco2(apnea_duration_s: float) -> float:
    """
    Return estimated arterial PaCO₂ (mmHg) after `apnea_duration_s` seconds of apnea.

    Args:
        apnea_duration_s: Duration of continuous apnea in seconds (≥ 0).

    Returns:
        Estimated PaCO₂ in mmHg.
    """
    if apnea_duration_s <= 0.0:
        return INITIAL_PACO2
    elif apnea_duration_s <= 60.0:
        return INITIAL_PACO2 + FIRST_MIN_RISE * (apnea_duration_s / 60.0)
    else:
        extra_minutes = (apnea_duration_s - 60.0) / 60.0
        return INITIAL_PACO2 + FIRST_MIN_RISE + SUBSEQUENT_RISE * extra_minutes


def co2_rise(apnea_duration_s: float) -> float:
    """
    Return the PaCO₂ rise (ΔPaCO₂) above baseline for a given apnea duration.

    Args:
        apnea_duration_s: Duration of continuous apnea in seconds.

    Returns:
        ΔPaCO₂ in mmHg (always ≥ 0).
    """
    return max(0.0, estimate_paco2(apnea_duration_s) - INITIAL_PACO2)


def should_alert_physiology(apnea_duration_s: float, threshold: float = ALERT_THRESHOLD) -> bool:
    """
    Return True if the physiological CO₂ model predicts a rise above `threshold`.

    This is the *guardrail* component of the fusion module.
    """
    return co2_rise(apnea_duration_s) > threshold


# ---------------------------------------------------------------------------
# Stateful tracker — used by the inference pipeline
# ---------------------------------------------------------------------------

@dataclass
class ApneaTracker:
    """
    Maintains the running apnea state across successive 10-second windows.

    Usage
    -----
    >>> tracker = ApneaTracker(window_sec=10)
    >>> for window_result in stream:
    ...     alert = tracker.update(ml_detected=window_result.is_apnea)
    ...     print(tracker.apnea_duration_s, tracker.get_co2_rise(), alert)
    """

    window_sec: float = 10.0         # duration of each detection window (s)
    alert_threshold: float = ALERT_THRESHOLD

    # Running state
    apnea_duration_s: float = field(default=0.0, init=False)
    _history: List[dict] = field(default_factory=list, init=False, repr=False)

    def update(self, ml_detected: bool) -> bool:
        """
        Advance the tracker by one window.

        Args:
            ml_detected: True if the ML model classified this window as apnea.

        Returns:
            alert_triggered: True if BOTH the ML model detected apnea AND the
                             physiological CO₂ model confirms a rise > threshold
                             (the fusion guardrail logic).
        """
        if ml_detected:
            self.apnea_duration_s += self.window_sec
        else:
            self.apnea_duration_s = 0.0

        paco2     = estimate_paco2(self.apnea_duration_s)
        rise      = co2_rise(self.apnea_duration_s)
        phys_ok   = rise > self.alert_threshold
        alert     = bool(ml_detected and phys_ok)   # fusion logic

        self._history.append({
            "apnea_duration_s": self.apnea_duration_s,
            "paco2_mmhg":       paco2,
            "co2_rise_mmhg":    rise,
            "ml_detected":      ml_detected,
            "phys_confirmed":   phys_ok,
            "alert":            alert,
        })
        return alert

    def get_paco2(self) -> float:
        """Current estimated PaCO₂ (mmHg)."""
        return estimate_paco2(self.apnea_duration_s)

    def get_co2_rise(self) -> float:
        """Current ΔPaCO₂ above baseline (mmHg)."""
        return co2_rise(self.apnea_duration_s)

    def trigger_alert(self) -> bool:
        """Whether the physiological threshold is currently exceeded (stateless call)."""
        return should_alert_physiology(self.apnea_duration_s, self.alert_threshold)

    def reset(self):
        """Reset tracker state (e.g. start of a new patient session)."""
        self.apnea_duration_s = 0.0
        self._history.clear()

    @property
    def history(self) -> List[dict]:
        return list(self._history)


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    print("PaCO₂ accumulation model")
    print("-" * 40)
    print(f"{'Time (s)':>10}  {'PaCO₂ (mmHg)':>14}  {'ΔPaCO₂ (mmHg)':>15}  {'Alert?':>7}")
    print("-" * 40)

    for t in [0, 10, 20, 30, 60, 90, 120, 180, 300]:
        paco2 = estimate_paco2(t)
        rise  = co2_rise(t)
        alert = should_alert_physiology(t)
        print(f"{t:>10}  {paco2:>14.2f}  {rise:>15.2f}  {'YES' if alert else 'no':>7}")

    print("\nApneaTracker simulation (window=10 s)")
    tracker = ApneaTracker(window_sec=10)
    pattern = [False, True, True, True, True, True, True, False, False, True, True]
    for i, detected in enumerate(pattern):
        alert = tracker.update(detected)
        print(
            f"  Window {i+1:02d}  ml={detected}  "
            f"apnea_dur={tracker.apnea_duration_s:.0f}s  "
            f"ΔCO₂={tracker.get_co2_rise():.2f}  "
            f"ALERT={'🔴' if alert else '🟢'}"
        )
