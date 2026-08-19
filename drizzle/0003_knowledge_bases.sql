CREATE TABLE `kb_chunks` (
	`id` varchar(36) NOT NULL,
	`kb_id` varchar(36) NOT NULL,
	`doc_name` varchar(200),
	`idx` int NOT NULL DEFAULT 0,
	`text` text,
	`embedding` json,
	CONSTRAINT `kb_chunks_id` PRIMARY KEY(`id`)
);
--> statement-breakpoint
CREATE TABLE `knowledge_bases` (
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
--> statement-breakpoint
CREATE INDEX `kb_chunks_kb_idx` ON `kb_chunks` (`kb_id`);--> statement-breakpoint
CREATE INDEX `kb_user_idx` ON `knowledge_bases` (`user_id`);