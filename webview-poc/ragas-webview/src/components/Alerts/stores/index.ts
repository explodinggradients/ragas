// Re-export everything from the context for compatibility
import type { Alert, CreateAlertParams } from '../context/AlertsContext';

export {
  type Alert,
  AlertPosition,
  AlertsProvider,
  AlertType,
  type CreateAlertParams,
  useAlerts,
} from '../context/AlertsContext';

// Legacy compatibility - this can be removed once all imports are updated
export const DEFAULT_ALERTS_STATE = {
  alerts: [],
};

export type AlertsStore = {
  alerts: Alert[];
  showAlert: ({ params }: { params: CreateAlertParams }) => string;
  dismissAlert: ({ id }: { id: string }) => void;
  clearAllAlerts: () => void;
  showSuccessAlert: ({ message, title, options }: { message: string; title?: string; options?: Partial<CreateAlertParams> }) => string;
  showErrorAlert: ({ message, title, options }: { message: string; title?: string; options?: Partial<CreateAlertParams> }) => string;
  showWarningAlert: ({ message, title, options }: { message: string; title?: string; options?: Partial<CreateAlertParams> }) => string;
  showInfoAlert: ({ message, title, options }: { message: string; title?: string; options?: Partial<CreateAlertParams> }) => string;
};

// Deprecated - use AlertsProvider and useAlerts hook instead
export const createAlertsStore = () => {
  console.warn('createAlertsStore is deprecated. Use AlertsProvider and useAlerts hook instead.');
  return {} as any;
};
