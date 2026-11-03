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
        self.lock = threading.Lock()

    def allow(self, key):
        """Records a hit and returns True, or returns False if the key is over its limit."""
        with self.lock:
            hits = self.hits[key]
            cutoff = self.clock() - self.window
            while hits and hits[0] <= cutoff:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(self.clock())
            return True
