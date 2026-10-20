"""Small timing policy for explicitly authorized compiled accumulation reruns."""
import math


def observe_update(self, seconds, previous):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("Update duration must be finite and positive")
    if not self.enabled:
        return max(previous, seconds)
    compiled = any(self.settings.get(key, False) for key in
                   ("model_compile", "muon_compile"))
    if compiled and not getattr(self, "_accumulation_cold_step_seen", False):
        self._accumulation_cold_step_seen = True
        self.metadata["cold_start_update_seconds"] = seconds
        self.metadata["next_update_initial_estimate_seconds"] = min(seconds, 30.0)
        return min(seconds, 30.0)
    self.recent_steps.append(seconds)
    return max(self.recent_steps)
