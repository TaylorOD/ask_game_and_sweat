-- Run once as the owner of a dedicated quota database before deployment.
-- Use separate databases/roles for each app; do not share the ReachOutt owner credential.
CREATE TABLE IF NOT EXISTS chat_budget_hits (
    namespace text NOT NULL,
    bucket text NOT NULL,
    subject text NOT NULL,
    expires_at timestamptz NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_budget_lookup
    ON chat_budget_hits (namespace, bucket, subject, expires_at);
CREATE INDEX IF NOT EXISTS chat_budget_expiry
    ON chat_budget_hits (namespace, expires_at);
-- Runtime role needs only CONNECT, USAGE on public, and SELECT, INSERT, DELETE
-- on this table. It does not need CREATE, UPDATE or access to another app's data.
