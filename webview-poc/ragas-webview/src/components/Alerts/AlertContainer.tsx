import { XIcon } from 'lucide-react';
import React from 'react';
import { useAlerts, AlertPosition } from '@/components/Alerts/context/AlertsContext';
import { Button } from '@/components/ui/button';
import { Alert } from './Alert';

const positionClasses = {
  [AlertPosition.TOP_CENTER]: 'top-10 left-1/2 -translate-x-1/2',
  [AlertPosition.TOP_LEFT]: 'top-10 left-4',
  [AlertPosition.TOP_RIGHT]: 'top-10 right-4',
  [AlertPosition.BOTTOM_CENTER]: 'bottom-10 left-1/2 -translate-x-1/2',
  [AlertPosition.BOTTOM_LEFT]: 'bottom-10 left-4',
  [AlertPosition.BOTTOM_RIGHT]: 'bottom-10 right-4',
};

export const AlertContainer: React.FC = () => {
  const { alerts, dismissAlert } = useAlerts();

  const groupedAlerts = alerts.reduce((acc, alert) => {
    const position = alert.position || AlertPosition.TOP_CENTER;
    if (!acc[position]) {
      acc[position] = [];
    }
    acc[position].push(alert);
    return acc;
  }, {} as Record<AlertPosition, typeof alerts>);

  return (
    <>
      {Object.entries(groupedAlerts).map(([position, positionAlerts]) => (
        <div
          key={position}
          className={`fixed z-50 flex min-w-[320px] max-w-[420px] flex-col gap-2 ${
            positionClasses[position as AlertPosition]
          }`}
        >
          {positionAlerts.map((alert) => (
            <Alert
              key={alert.id}
              variant={alert.type}
              className="animate-in fade-in slide-in-from-top-4"
            >
              <div className="flex w-full items-center justify-between">
                <div className="flex-1 text-foreground">
                  {alert.title && (
                    <Alert.Title className="font-semibold">
                      {alert.title}
                    </Alert.Title>
                  )}
                  <Alert.Description>{alert.message}</Alert.Description>
                </div>
                {alert.isDismissible !== false && (
                  <Button
                    variant="ghost"
                    size="icon"
                    className="ml-4 size-8"
                    onClick={() => dismissAlert(alert.id)}
                  >
                    <XIcon className="size-4" />
                  </Button>
                )}
              </div>
            </Alert>
          ))}
        </div>
      ))}
    </>
  );
};
