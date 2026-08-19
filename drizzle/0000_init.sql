CREATE TABLE `audit_log` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36),
	`event` varchar(60) NOT NULL,
	`detail` json,
	`ts` timestamp DEFAULT (now()),
	CONSTRAINT `audit_log_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `projects` (
	`id` varchar(36) NOT NULL,
	`user_id` varchar(36) NOT NULL,
	`lab` varchar(40) NOT NULL,
	`name` varchar(160) NOT NULL,
	`config` json,
	`shared` boolean NOT NULL DEFAULT false,
	`created_at` timestamp DEFAULT (now()),
	`updated_at` timestamp DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `projects_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `providers` (
	`id` varchar(36) NOT NULL,
	`provider` varchar(40) NOT NULL,
	`label` varchar(120),
	`base_url` varchar(255) NOT NULL,
	`api_key_enc` text,
	`default_model` varchar(120),
	`enabled` boolean NOT NULL DEFAULT true,
	`created_at` timestamp DEFAULT (now()),
	`updated_at` timestamp DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `providers_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `rate_limits` (
	`id` varchar(140) NOT NULL,
	`scope` varchar(40) NOT NULL,
	`window_start` bigint NOT NULL,
	`count` int NOT NULL DEFAULT 0,
	`updated_at` timestamp DEFAULT (now()) ON UPDATE CURRENT_TIMESTAMP,
	CONSTRAINT `rate_limits_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `sessions` (
	`token` varchar(64) NOT NULL,
	`user_id` varchar(36) NOT NULL,
	`expires_at` timestamp NOT NULL,
	`created_at` timestamp DEFAULT (now()),
	CONSTRAINT `sessions_token` PRIMARY KEY(`token`)
);
--> statement-breakpoint
CREATE TABLE `usage` (
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
--> statement-breakpoint
CREATE TABLE `users` (
	`id` varchar(36) NOT NULL,
	`email` varchar(255) NOT NULL,
	`name` varchar(120),
	`password_hash` varchar(255) NOT NULL,
	`role` enum('admin','student') NOT NULL DEFAULT 'student',
	`status` enum('pending','active','suspended') NOT NULL DEFAULT 'pending',
	`monthly_token_limit` int,
	`created_at` timestamp DEFAULT (now()),
	CONSTRAINT `users_id` PRIMARY KEY(`id`),
	CONSTRAINT `users_email_unique` UNIQUE(`email`)
);
--> statement-breakpoint
CREATE INDEX `audit_event_ts_idx` ON `audit_log` (`event`,`ts`);--> statement-breakpoint
CREATE INDEX `usage_user_ts_idx` ON `usage` (`user_id`,`ts`);