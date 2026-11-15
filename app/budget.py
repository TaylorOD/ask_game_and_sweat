"""Atomic rolling quotas shared across restarts and workers via PostgreSQL."""
import hashlib
import hmac
import os
import threading
import time
from collections import deque

import psycopg


class BudgetUnavailable(Exception):
    pass


class MemoryBudget:
    """Local development only. Set REQUIRE_PERSISTENT_LIMITS=true on hosts."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.hits = {}
        self.lock = threading.Lock()

    def reserve(self, rules):
        with self.lock:
            now = self.clock()
            for key, hits in list(self.hits.items()):
                while hits and hits[0] <= now:
                    hits.popleft()
                if not hits:
                    del self.hits[key]
            for bucket, subject, limit, window in rules:
                if len(self.hits.get((bucket, subject), ())) >= limit:
                    return bucket
            for bucket, subject, _, window in rules:
                self.hits.setdefault((bucket, subject), deque()).append(now + window)
            return None


class PostgresBudget:
    def __init__(self, url, namespace):
        self.url = url
        self.namespace = namespace
        # Stable across processes; never persist visitor IP addresses.
        self.secret = url.encode()
        self.lock_id = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big", signed=True)

    def reserve(self, rules):
        try:
            with psycopg.connect(self.url, connect_timeout=5,
                                 options="-c statement_timeout=5000 -c lock_timeout=2000") as conn:
                # One transaction admits all buckets or none. The shared lock also
                # prevents separate workers from taking the same last slot.
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (self.lock_id,))
                now = conn.execute("SELECT clock_timestamp()").fetchone()[0]
                conn.execute("DELETE FROM chat_budget_hits WHERE namespace=%s AND expires_at <= %s",
                             (self.namespace, now))
                subjects = []
                for bucket, subject, limit, window in rules:
                    digest = hmac.new(self.secret, subject.encode(), hashlib.sha256).hexdigest()
                    subjects.append((bucket, digest, window))
                    count = conn.execute(
                        "SELECT count(*) FROM chat_budget_hits WHERE namespace=%s AND bucket=%s "
                        "AND subject=%s AND expires_at > %s",
                        (self.namespace, bucket, digest, now)).fetchone()[0]
                    if count >= limit:
                        return bucket
                for bucket, digest, window in subjects:
                    conn.execute(
                        "INSERT INTO chat_budget_hits (namespace,bucket,subject,expires_at) "
                        "VALUES (%s,%s,%s,%s + %s * interval '1 second')",
                        (self.namespace, bucket, digest, now, window))
            return None
        except psycopg.Error:
            # Do not leak a DSN, password, SQL statement or provider body.
            raise BudgetUnavailable() from None


def create_budget(namespace):
    url = os.getenv("RATE_LIMIT_DATABASE_URL")
    if url:
        return PostgresBudget(url, namespace)
    if os.getenv("REQUIRE_PERSISTENT_LIMITS", "false").lower() == "true":
        raise RuntimeError("RATE_LIMIT_DATABASE_URL is required for persistent chat limits")
    return MemoryBudget()
