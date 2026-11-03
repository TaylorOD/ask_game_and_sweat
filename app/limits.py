import threading
import time
from collections import defaultdict, deque


class RateLimiter:
    """Allows `limit` hits per key in a rolling window. Lives in memory, so it
    resets on restart and only works with a single server process."""

    def __init__(self, limit, window_seconds, clock=time.monotonic):
        self.limit = limit
        self.window = window_seconds
        self.clock = clock
        self.hits = defaultdict(deque)
        self.last_sweep = clock()
        self.lock = threading.Lock()

    def _sweep(self, now):
        # Forget visitors with nothing left in the window, or memory grows with every new visitor
        if now - self.last_sweep < self.window:
            return
        self.last_sweep = now
        for key in [k for k, hits in self.hits.items() if not hits or hits[-1] <= now - self.window]:
            del self.hits[key]

    def allow(self, key):
        """Records a hit and returns True, or returns False if the key is over its limit."""
        with self.lock:
            now = self.clock()
            self._sweep(now)
            hits = self.hits[key]
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            return True
