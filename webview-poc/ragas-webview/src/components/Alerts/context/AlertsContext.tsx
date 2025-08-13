import { createContext, type ReactNode, useCallback, useContext, useReducer } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { logger } from '@/helpers/logger.ts';

// Alert types and position enums
export enum AlertType {
  SUCCESS = 'success',
  ERROR = 'error',
  WARNING = 'warning',
  INFO = 'info',
}

export enum AlertPosition {
  TOP_LEFT = 'top-left',
  TOP_CENTER = 'top-center',
  TOP_RIGHT = 'top-right',
  BOTTOM_LEFT = 'bottom-left',
  BOTTOM_CENTER = 'bottom-center',
  BOTTOM_RIGHT = 'bottom-right',
}

// Alert model
export type Alert = {
  id: string;
  type: AlertType;
  message: string;
  title?: string;
  position: AlertPosition;
  timeout?: number;
  isDismissible?: boolean;
  metadata?: Record<string, unknown>;
};

// Create alert params
export type CreateAlertParams = Omit<Alert, 'id'>;

// Context state
type AlertsState = {
  alerts: Alert[];
};

// Context actions
type AlertsActions = {
  showAlert: (params: CreateAlertParams) => string;
  dismissAlert: (id: string) => void;
  clearAllAlerts: () => void;
  showSuccessAlert: (message: string, title?: string, options?: Partial<CreateAlertParams>) => string;
  showErrorAlert: (message: string, title?: string, options?: Partial<CreateAlertParams>) => string;
  showWarningAlert: (message: string, title?: string, options?: Partial<CreateAlertParams>) => string;
  showInfoAlert: (message: string, title?: string, options?: Partial<CreateAlertParams>) => string;
};

type AlertsContextType = AlertsState & AlertsActions;

// Reducer actions
type AlertsReducerAction =
  | { type: 'SHOW_ALERT'; payload: Alert }
  | { type: 'DISMISS_ALERT'; payload: { id: string } }
  | { type: 'CLEAR_ALL_ALERTS' };

// Initial state
const initialState: AlertsState = {
  alerts: [],
};

// Reducer
function alertsReducer(state: AlertsState, action: AlertsReducerAction): AlertsState {
  switch (action.type) {
    case 'SHOW_ALERT':
      return {
        ...state,
        alerts: [...state.alerts, action.payload],
      };
    case 'DISMISS_ALERT':
      return {
        ...state,
        alerts: state.alerts.filter((alert) => alert.id !== action.payload.id),
      };
    case 'CLEAR_ALL_ALERTS':
      return {
        ...state,
        alerts: [],
      };
    default:
      return state;
  }
}

// Helper function to generate a unique ID
const generateId = (): string => {
  return uuidv4();
};

// Create context
const AlertsContext = createContext<AlertsContextType | undefined>(undefined);

// Provider component
type AlertsProviderProps = {
  children: ReactNode;
};

export function AlertsProvider({ children }: AlertsProviderProps) {
  const [state, dispatch] = useReducer(alertsReducer, initialState);

  const showAlert = useCallback((params: CreateAlertParams): string => {
    logger.debug('[AlertsContext] Showing alert:', params);

    const id = generateId();
    const newAlert: Alert = {
      ...params,
      id,
      position: params.position || AlertPosition.TOP_CENTER,
      isDismissible: params.isDismissible ?? true,
    };

    dispatch({ type: 'SHOW_ALERT', payload: newAlert });

    // Set timeout for auto-dismissal if timeout is provided
    if (params.timeout && params.timeout > 0) {
      setTimeout(() => {
        dispatch({ type: 'DISMISS_ALERT', payload: { id } });
        logger.debug(`[AlertsContext] Auto-dismissed alert ${id} after ${params.timeout}ms`);
      }, params.timeout);
    }

    return id;
  }, []);

  const dismissAlert = useCallback((id: string) => {
    logger.debug(`[AlertsContext] Dismissing alert with id: ${id}`);
    dispatch({ type: 'DISMISS_ALERT', payload: { id } });
  }, []);

  const clearAllAlerts = useCallback(() => {
    logger.debug('[AlertsContext] Clearing all alerts');
    dispatch({ type: 'CLEAR_ALL_ALERTS' });
  }, []);

  // Convenience methods for specific alert types
  const showSuccessAlert = useCallback((
    message: string,
    title?: string,
    options?: Partial<CreateAlertParams>,
  ): string => {
    return showAlert({
      type: AlertType.SUCCESS,
      message,
      title,
      position: AlertPosition.TOP_CENTER,
      timeout: 5000, // Default timeout for success alerts
      ...options,
    });
  }, [showAlert]);

  const showErrorAlert = useCallback((
    message: string,
    title?: string,
    options?: Partial<CreateAlertParams>,
  ): string => {
    return showAlert({
      type: AlertType.ERROR,
      message,
      title,
      position: AlertPosition.TOP_CENTER,
      timeout: 0, // No default timeout for error alerts
      ...options,
    });
  }, [showAlert]);

  const showWarningAlert = useCallback((
    message: string,
    title?: string,
    options?: Partial<CreateAlertParams>,
  ): string => {
    return showAlert({
      type: AlertType.WARNING,
      message,
      title,
      position: AlertPosition.TOP_CENTER,
      timeout: 7000, // Default timeout for warning alerts
      ...options,
    });
  }, [showAlert]);

  const showInfoAlert = useCallback((
    message: string,
    title?: string,
    options?: Partial<CreateAlertParams>,
  ): string => {
    return showAlert({
      type: AlertType.INFO,
      message,
      title,
      position: AlertPosition.TOP_CENTER,
      timeout: 5000, // Default timeout for info alerts
      ...options,
    });
  }, [showAlert]);

  const contextValue: AlertsContextType = {
    alerts: state.alerts,
    showAlert,
    dismissAlert,
    clearAllAlerts,
    showSuccessAlert,
    showErrorAlert,
    showWarningAlert,
    showInfoAlert,
  };

  return (
    <AlertsContext.Provider value={contextValue}>
      {children}
    </AlertsContext.Provider>
  );
}

// Hook to use the alerts context
export function useAlerts(): AlertsContextType {
  const context = useContext(AlertsContext);
  if (context === undefined) {
    throw new Error('useAlerts must be used within an AlertsProvider');
  }
  return context;
}
