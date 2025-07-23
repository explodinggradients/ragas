import type { Wretch } from 'wretch';
import wretch from 'wretch';
import { logger } from '../logger';

/**
 * Configuration options for the ApiClient
 */
export type ApiClientOptions = {
  /**
   * Default headers to include with every request
   */
  headers?: Record<string, string>;

  /**
   * Default fetch options to include with every request
   */
  fetchOptions?: RequestInit;
};

/**
 * A client for making API requests using wretch
 */
export class ApiClient {
  protected readonly wretchInstance: Wretch<unknown, unknown, undefined>;

  /**
   * Create a new ApiClient
   *
   * @param baseUrl The base URL for all API requests
   * @param options Configuration options
   */
  constructor(
    private readonly baseUrl: string,
    options?: ApiClientOptions,
  ) {
    // Create and configure the wretch instance
    this.wretchInstance = wretch(baseUrl)
      .options(options?.fetchOptions || {})
      .headers(options?.headers || {});
  }

  /**
   * Get the base URL
   */
  getBaseUrl(): string {
    return this.baseUrl;
  }

  /**
   * Add headers to the request
   *
   * @param headers Headers to add
   * @returns A new instance with the headers applied
   */
  withHeaders(headers: Record<string, string>): ApiClient {
    const newClient = new ApiClient(this.baseUrl);
    newClient.wretchInstance.headers(headers);
    return newClient;
  }

  /**
   * Set authorization header
   *
   * @param token The authorization token
   * @returns A new instance with the authorization header applied
   */
  withAuth(token: string): ApiClient {
    return this.withHeaders({ Authorization: `Bearer ${token}` });
  }

  /**
   * Make a GET request
   *
   * @param url The URL to request (will be appended to baseUrl)
   * @param options Additional fetch options
   * @returns A promise that resolves to the response data
   */
  get<T>(url: string, options?: RequestInit): Promise<T> {
    return this.wretchInstance
      .url(url)
      .options(options || {})
      .get()
      .json<T>();
  }

  /**
   * Make a POST request
   *
   * @param url The URL to request (will be appended to baseUrl)
   * @param body The request body
   * @param options Additional fetch options
   * @returns A promise that resolves to the response data
   */
  post<T, B extends object>(url: string, body: B, options?: RequestInit): Promise<T> {
    const request = this.wretchInstance
      .url(url)
      .options(options || {});

    return request.json(body).post().json<T>();
  }

  /**
   * Make a PUT request
   *
   * @param url The URL to request (will be appended to baseUrl)
   * @param body The request body
   * @param options Additional fetch options
   * @returns A promise that resolves to the response data
   */
  put<T, B extends object>(url: string, body: B, options?: RequestInit): Promise<T> {
    const request = this.wretchInstance
      .url(url)
      .options(options || {});

    return request.json(body).put().json<T>();
  }

  /**
   * Make a PATCH request
   *
   * @param url The URL to request (will be appended to baseUrl)
   * @param body The request body
   * @param options Additional fetch options
   * @returns A promise that resolves to the response data
   */
  patch<T, B extends object>(url: string, body: B, options?: RequestInit): Promise<T> {
    const request = this.wretchInstance
      .url(url)
      .options(options || {});

    return request.json(body).patch().json<T>();
  }

  /**
   * Make a DELETE request
   *
   * @param url The URL to request (will be appended to baseUrl)
   * @param options Additional fetch options
   * @returns A promise that resolves to the response data
   */
  delete<T>(url: string, options?: RequestInit): Promise<T> {
    return this.wretchInstance
      .url(url)
      .options(options || {})
      .delete()
      .json<T>();
  }

  /**
   * Make a HEAD request
   *
   * @param url The URL to request (will be appended to baseUrl)
   * @param options Additional fetch options
   * @returns A promise that resolves to the response
   */
  head(url: string, options?: RequestInit): Promise<Response> {
    return this.wretchInstance
      .url(url)
      .options(options || {})
      .head()
      .res();
  }

  /**
   * Make a OPTIONS request
   *
   * @param url The URL to request (will be appended to baseUrl)
   * @param fetchOptions
   * @returns A promise that resolves to the response
   */
  options<T>(url: string, fetchOptions?: RequestInit): Promise<T> {
    return this.wretchInstance
      .url(url)
      .options(fetchOptions || {})
      .opts()
      .json<T>();
  }

  /**
   * Handle common error responses
   *
   * @param callback The callback to execute after error handling
   * @returns The response or throws an error
   */
  withErrorHandling<T>(callback: () => Promise<T>): Promise<T> {
    return callback()
      .catch((error) => {
        logger.error('API request failed:', error);
        throw error;
      });
  }
}
