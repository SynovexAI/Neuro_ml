CREATE TABLE `rag_experiments` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36) NOT NULL,
	`label` varchar(160) NOT NULL,
	`dataset` varchar(200),
	`question` text,
	`config` json,
	`metrics` json,
	`chunk_count` int NOT NULL DEFAULT 0,
	`latency_ms` int NOT NULL DEFAULT 0,
	`ts` timestamp DEFAULT (now()),
	CONSTRAINT `rag_experiments_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE INDEX `rag_exp_user_ts_idx` ON `rag_experiments` (`user_id`,`ts`);
