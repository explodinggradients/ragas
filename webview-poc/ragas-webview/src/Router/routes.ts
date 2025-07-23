/**
 * Common routes used throughout the application.
 * Add any new routes here and reference them using this object.
 * DO NOT hardcode route paths anywhere else in the app.
 */

export const COMMON_ROUTES = {
  root: '/',
  datasets: '/datasets',
  experiments: '/experiments',
} as const;

/**
 * Common query parameter keys used throughout the application
 */
export const COMMON_QUERY_PARAMS = {
  // Add query params here as needed
} as const;