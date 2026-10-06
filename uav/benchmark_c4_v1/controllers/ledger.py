"""EvaluationLedger: counts ALL objective evaluations. No free evaluations allowed."""

from __future__ import annotations


class EvaluationLedger:
    """Central FE counter. Every objective evaluation MUST go through this."""

    def __init__(self, max_fe: int):
        self.max_fe = max_fe
        self.count = 0

    def charge(self, n: int) -> bool:
        """Charge n evaluations. Returns True if budget allows, False if exceeded."""
        if self.count + n > self.max_fe:
            return False
        self.count += n
        return True

    def remaining(self) -> int:
        return max(0, self.max_fe - self.count)

    @property
    def exhausted(self) -> bool:
        return self.count >= self.max_fe

    def reset(self):
        self.count = 0

    def summary(self) -> dict:
        return {"total_fe": self.count, "max_fe": self.max_fe,
                "remaining": self.remaining(), "exhausted": self.exhausted}
