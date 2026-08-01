-- ────────────────────────────────────────────────────────────────────────────
-- DELTA for an EXISTING database that already has users/sessions/providers/
-- projects (i.e. the DB created via `drizzle-kit push` before migrations
-- existed). It adds only the NEW cost-control / ops objects.
--
--   • FRESH database  → don't use this; run `npm run db:migrate` instead
--                        (applies drizzle/0000_init.sql — the full schema).
--   • EXISTING database → run this once (or just run `npm run db:push`, which
--                        auto-diffs and applies the same delta).
--
-- Safe to re-run: every statement uses IF NOT EXISTS (supported by TiDB).
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS `usage` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36) NOT NULL,
	`lab` varchar(40),
	`model` varchar(120),
	`prompt_tokens` int NOT NULL DEFAULT 0,
	`completion_tokens` int NOT NULL DEFAULT 0,
	`total_tokens` int NOT NULL DEFAULT 0,
	`estimated` boolean NOT NULL DEFAULT false,
	`ts` timestamp DEFAULT (now()),
	CONSTRAINT `usage_id` PRIMARY KEY(`id`)
);

CREATE TABLE IF NOT EXISTS `rate_limits` (
	`id` varchar(140) NOT NULL,
	`scope` varchar(40) NOT NULL,
	`window_start` bigint NOT NULL,
	`count` int NOT NULL DEFAULT 0,
	`updated_at` timestamp DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `rate_limits_id` PRIMARY KEY(`id`)
);

CREATE TABLE IF NOT EXISTS `audit_log` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36),
	`event` varchar(60) NOT NULL,
	`detail` json,
	`ts` timestamp DEFAULT (now()),
	CONSTRAINT `audit_log_id` PRIMARY KEY(`id`)
);

CREATE TABLE IF NOT EXISTS `mcp_servers` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36),
	`name` varchar(80) NOT NULL,
	`transport` enum('http','sse','stdio') NOT NULL DEFAULT 'http',
	`url` varchar(500),
	`command` varchar(500),
	`auth_type` enum('none','apikey','bearer','oauth') NOT NULL DEFAULT 'none',
	`header_name` varchar(80),
	`env_name` varchar(80),
	`secret_enc` text,
	`enabled` boolean NOT NULL DEFAULT true,
	`tools` json,
	`status` varchar(40) DEFAULT 'configured',
	`created_at` timestamp DEFAULT (now()),
	`updated_at` timestamp DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `mcp_servers_id` PRIMARY KEY(`id`)
);

CREATE TABLE IF NOT EXISTS `agent_runs` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36) NOT NULL,
	`agent_name` varchar(120),
	`agent_type` varchar(24),
	`runtime` varchar(16),
	`provider` varchar(40),
	`model` varchar(120),
	`iterations` int NOT NULL DEFAULT 0,
	`tool_calls` json,
	`tool_call_count` int NOT NULL DEFAULT 0,
	`prompt_tokens` int NOT NULL DEFAULT 0,
	`completion_tokens` int NOT NULL DEFAULT 0,
	`total_tokens` int NOT NULL DEFAULT 0,
	`latency_ms` int NOT NULL DEFAULT 0,
	`outcome` varchar(24),
	`error_msg` varchar(300),
	`ts` timestamp DEFAULT (now()),
	CONSTRAINT `agent_runs_id` PRIMARY KEY(`id`)
);

CREATE TABLE IF NOT EXISTS `knowledge_bases` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36) NOT NULL,
	`name` varchar(120) NOT NULL,
	`status` varchar(24) NOT NULL DEFAULT 'empty',
	`doc_count` int NOT NULL DEFAULT 0,
	`chunk_count` int NOT NULL DEFAULT 0,
	`emb_model` varchar(120),
	`emb_meta` json,
	`created_at` timestamp DEFAULT (now()),
	`updated_at` timestamp DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `knowledge_bases_id` PRIMARY KEY(`id`)
);

CREATE TABLE IF NOT EXISTS `kb_chunks` (
	`id` varchar(36) NOT NULL,
	`kb_id` varchar(36) NOT NULL,
	`doc_name` varchar(200),
	`idx` int NOT NULL DEFAULT 0,
	`text` text,
	`embedding` json,
	CONSTRAINT `kb_chunks_id` PRIMARY KEY(`id`)
);

ALTER TABLE `users` ADD COLUMN IF NOT EXISTS `monthly_token_limit` int;
ALTER TABLE `mcp_servers` ADD COLUMN IF NOT EXISTS `user_id` varchar(36);

CREATE INDEX IF NOT EXISTS `usage_user_ts_idx` ON `usage` (`user_id`,`ts`);
CREATE INDEX IF NOT EXISTS `audit_event_ts_idx` ON `audit_log` (`event`,`ts`);
CREATE INDEX IF NOT EXISTS `agent_runs_user_ts_idx` ON `agent_runs` (`user_id`,`ts`);
CREATE INDEX IF NOT EXISTS `kb_user_idx` ON `knowledge_bases` (`user_id`);
CREATE INDEX IF NOT EXISTS `kb_chunks_kb_idx` ON `kb_chunks` (`kb_id`);

-- ── Workroom: publish agents + deploy to channels (Telegram / API / widget) ──
ALTER TABLE `projects` ADD COLUMN IF NOT EXISTS `published` boolean NOT NULL DEFAULT false;
CREATE TABLE IF NOT EXISTS `channels` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36) NOT NULL,
	`project_id` varchar(36) NOT NULL,
	`type` varchar(24) NOT NULL,
	`secret_enc` text,
	`enabled` boolean NOT NULL DEFAULT true,
	`created_at` timestamp DEFAULT (now()),
	CONSTRAINT `channels_id` PRIMARY KEY(`id`)
);
CREATE INDEX IF NOT EXISTS `channels_project_idx` ON `channels` (`project_id`);
