import { VITE_DEFAULT_BUNDLED_SERVER_HOST, VITE_DEFAULT_BUNDLED_SERVER_PORT } from '@/constants/env';

type QueryParams = Record<string, string>;

type UrlInfo = {
  protocol: string;
  host: string;
  origin: string;
};

/**
 * Get the absolute URL information for the current page in the browser
 * Handles various edge cases around protocols and hosts
 */
export function getWebsiteUrl(): UrlInfo {
  // Get protocol from location, ensuring proper format
  let protocol = window.location.protocol;
  protocol = protocol.endsWith(':') ? protocol : `${protocol}:`;

  // Get host from location, removing any trailing slashes
  const host = window.location.host.replace(/\/$/, '');

  // For most cases, we can use window.location.origin
  // But we reconstruct it to ensure consistent formatting with protocol handling
  const origin = `${protocol}//${host}`;

  return {
    protocol,
    host,
    origin,
  };
}

/**
 * Optional: Get specific URL parts safely
 */
export function getUrlParts() {
  return {
    pathname: window.location.pathname,
    search: window.location.search,
    hash: window.location.hash,
    hostname: window.location.hostname,
    port: window.location.port,
  };
}

/**
 * Get the API base URL
 * - In dev mode: Uses VITE environment variables if available
 * - In CLI mode: Uses same origin as the website
 */
export function getApiUrl(): string {
  // If environment variables are set (dev mode), use them
  if (VITE_DEFAULT_BUNDLED_SERVER_HOST && VITE_DEFAULT_BUNDLED_SERVER_PORT) {
    return `http://${VITE_DEFAULT_BUNDLED_SERVER_HOST}:${VITE_DEFAULT_BUNDLED_SERVER_PORT}`;
  }

  // Otherwise use same origin (CLI mode)
  const webUrl = getWebsiteUrl();
  return webUrl.origin;
}

export function buildProjectUrlForBrowser(urlPath: string, queryParams?: QueryParams) {
  const webUrl = getWebsiteUrl();
  const baseUrl = `${webUrl.origin}${urlPath}`;

  if (!queryParams || Object.keys(queryParams).length === 0) {
    return baseUrl;
  }

  const searchParams = new URLSearchParams();
  Object.entries(queryParams).forEach(([key, value]) => {
    searchParams.append(key, value);
  });

  const queryString = searchParams.toString();
  return queryString ? `${baseUrl}?${queryString}` : baseUrl;
}
