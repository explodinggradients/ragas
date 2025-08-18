import type { ApiClientOptions } from './api-client';
import { getApiUrl } from '@/helpers/urls';
import { ApiClient } from './api-client';

/**
 * API response status type
 */
export type BackendApiResponseStatus = 'success' | 'error';

/**
 * Success response interface
 */
export type BackendSuccessResponse<T = unknown> = {
  status: 'success';
  status_code: number;
  message: string;
  data: T | null;
};

/**
 * Error response interface
 */
export type BackendErrorResponse = {
  status: 'error';
  status_code: number;
  message: string;
  data: null;
  debug_error_info?: {
    error_type?: string;
    error_message?: string;
    debug_message?: string;
    stack?: string;
  };
  error_details: Record<string, unknown> | undefined;
};

export type BackendApiResponseType<T> = BackendSuccessResponse<T> | BackendErrorResponse;

/**
 * Custom API error with status code and debug information
 */
export class BackendApiError extends Error {
  readonly statusCode: number;
  readonly debugMessage?: string;
  readonly originalError?: Error;
  readonly errorDetails?: Record<string, unknown>;

  constructor(
    message: string,
    statusCode = 400,
    debugMessage?: string,
    originalError?: Error,
    errorDetails?: Record<string, unknown>,
  ) {
    super(message);
    this.name = 'BackendApiError';
    this.statusCode = statusCode;
    this.debugMessage = debugMessage;
    this.originalError = originalError;
    this.errorDetails = errorDetails;
  }
}

/**
 * Backend API Client
 */
export class BackendApiClient extends ApiClient {
  constructor(
    baseUrl: string,
    options?: ApiClientOptions,
  ) {
    super(baseUrl, options);
  }

  /**
   * Override to handle standardized API responses
   */
  override async get<T>(url: string, options?: RequestInit): Promise<T> {
    return super.get<BackendApiResponseType<T>>(url, options).then(this.handleApiResponse<T>);
  }

  /**
   * Override to handle standardized API responses
   */
  override async post<T, B extends object>(url: string, body: B, options?: RequestInit): Promise<T> {
    return super.post<BackendApiResponseType<T>, B>(url, body, options).then(this.handleApiResponse<T>);
  }

  /**
   * Override to handle standardized API responses
   */
  override async put<T, B extends object>(url: string, body: B, options?: RequestInit): Promise<T> {
    return super.put<BackendApiResponseType<T>, B>(url, body, options).then(this.handleApiResponse<T>);
  }

  /**
   * Override to handle standardized API responses
   */
  override async patch<T, B extends object>(url: string, body: B, options?: RequestInit): Promise<T> {
    return super.patch<BackendApiResponseType<T>, B>(url, body, options).then(this.handleApiResponse<T>);
  }

  /**
   * Override to handle standardized API responses
   */
  override async delete<T>(url: string, options?: RequestInit): Promise<T> {
    return super.delete<BackendApiResponseType<T>>(url, options).then(this.handleApiResponse<T>);
  }

  /**
   * Override to handle standardized API responses
   */
  override async options<T>(url: string, fetchOptions?: RequestInit): Promise<T> {
    return super.options<BackendApiResponseType<T>>(url, fetchOptions).then(this.handleApiResponse<T>);
  }

  /**
   * Process the API response and extract data or throw an error
   * @param response The API response
   * @returns The data from a success response
   * @throws BackendApiError for error responses
   */
  private handleApiResponse<T>(response: BackendApiResponseType<T>): T {
    if (response.status === 'success') {
      // Return the data directly for success responses
      return response.data as T;
    } else {
      // Throw an BackendApiError for error responses
      throw new BackendApiError(
        response.message,
        response.status_code,
        response.debug_error_info?.debug_message,
        response.debug_error_info?.error_message ? new Error(response.debug_error_info.error_message) : undefined,
      );
    }
  }
}

/**
 * Creates a backend API client
 *
 * @param baseUrl The base URL for the backend API
 * @param options Additional API client options
 * @returns BackendApiClient instance
 */
export const createBackendApiClient = (
  baseUrl: string,
  options?: ApiClientOptions,
): BackendApiClient => {
  return new BackendApiClient(baseUrl, options);
};

/**
 * Backend API client instance
 */
export const backendApiClient = createBackendApiClient(
  getApiUrl(),
);
