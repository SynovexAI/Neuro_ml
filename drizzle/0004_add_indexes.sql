-- Performance indexes for hot per-user / time-range query paths.
-- Apply once (or run `npm run db:push` which syncs these from schema.ts).
-- MySQL has no CREATE INDEX IF NOT EXISTS; skip any that already exist.
CREATE INDEX projects_user_idx ON projects (user_id);
CREATE INDEX mcp_user_idx ON mcp_servers (user_id);
CREATE INDEX usage_ts_idx ON usage (ts);
