/* eslint-disable ts/no-explicit-any */

import { CURRENT_LOG_LEVEL, IS_PROD, LOG_LEVELS, type LogLevelKind } from '@/constants/env.ts';

export const canLog = (level: LogLevelKind) =>
  LOG_LEVELS.indexOf(level) >= LOG_LEVELS.indexOf(CURRENT_LOG_LEVEL);

export const logger = (() => {
  const levelEmojis: Record<string, string> = {
    trace: '📍 [TRACE][development-only]',
    debug: '🔍 [DEBUG][development-only]',
    info: 'ℹ️ [INFO]',
    warn: '⚠️ [WARN]',
    error: '❌ [ERROR]',
  };

  const createLoggerMethod = (
    level: LogLevelKind,
    consoleMethod: (...args: any[]) => void,
  ) =>
    (...args: any) => {
      if (canLog(level)) {
        consoleMethod(`${levelEmojis[level]}`, ...args);
      }
    };

  return {
    trace: createLoggerMethod('trace', console.trace),
    debug: createLoggerMethod('debug', console.debug),
    info: createLoggerMethod('info', console.log),
    warn: createLoggerMethod('warn', console.warn),
    error: createLoggerMethod('error', console.error),
  };
})();

export function printBoundary(char = '═', length = 70, allowInProd = false) {
  if (IS_PROD && !allowInProd) {
    return;
  }

  let output = char;

  for (let i = 0; i < length; i += 1) {
    output += char;
  }

  console.info(output);
}
