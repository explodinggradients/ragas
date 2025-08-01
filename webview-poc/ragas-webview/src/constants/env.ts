import { z } from 'zod';

export const LogLevel = z.enum(['trace', 'debug', 'info', 'warn', 'error']);

export type LogLevelKind = z.infer<typeof LogLevel>;

export const LOG_LEVELS: LogLevelKind[] = ['trace', 'debug', 'info', 'warn', 'error'];

export const NodeEnv = process.env.NODE_ENV ?? 'production'; // by default mark it as prod

export const IS_PROD = NodeEnv === 'production';

export const IS_DEV = NodeEnv === 'development';

export const CURRENT_LOG_LEVEL = IS_DEV ? 'trace' : 'info';

// Bundled server configuration from environment variables
export const VITE_DEFAULT_BUNDLED_SERVER_HOST = import.meta.env.VITE_DEFAULT_BUNDLED_SERVER_HOST;
export const VITE_DEFAULT_BUNDLED_SERVER_PORT = import.meta.env.VITE_DEFAULT_BUNDLED_SERVER_PORT;
