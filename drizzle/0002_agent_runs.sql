CREATE TABLE `agent_runs` (
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
--> statement-breakpoint
CREATE INDEX `agent_runs_user_ts_idx` ON `agent_runs` (`user_id`,`ts`);