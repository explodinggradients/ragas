// API Configuration
const API_HOST = import.meta.env.VITE_API_HOST || 'localhost';
const API_PORT = import.meta.env.VITE_API_PORT || 8000;

export const API_BASE_URL = `http://${API_HOST}:${API_PORT}`;
export const API_ENDPOINTS = {
  DATASETS: `${API_BASE_URL}/api/datasets`,
  HEALTH: `${API_BASE_URL}/api/health`,
} as const;