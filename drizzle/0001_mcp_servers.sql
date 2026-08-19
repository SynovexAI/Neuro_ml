CREATE TABLE `mcp_servers` (
	`id` varchar(36) NOT NULL,
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
