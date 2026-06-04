from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class Profiler:
    breakdown: dict[str, float] = field(default_factory=dict)

    @contextmanager
    def span(self, label: str):
        t0 = time.monotonic()
        yield
        self.breakdown[label] = round((time.monotonic() - t0) * 1000, 1)

    @property
    def total_ms(self) -> float:
        return round(sum(self.breakdown.values()), 1)
