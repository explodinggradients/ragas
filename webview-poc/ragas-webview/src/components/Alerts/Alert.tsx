import { cva, type VariantProps } from 'class-variance-authority';
import { AlertCircle, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import * as React from 'react';
import { cn } from '@/lib/utils';
import { AlertType } from './stores';

const alertVariants = cva(
  'relative flex w-full items-center rounded-full border px-4 py-3 text-sm text-foreground shadow-md',
  {
    variants: {
      variant: {
        [AlertType.ERROR]: 'border-red-100 bg-background/100 dark:border-red-400',
        [AlertType.WARNING]: 'border-yellow-100 bg-background/100 dark:border-yellow-400',
        [AlertType.SUCCESS]: 'border-green-100 bg-background/100 dark:border-green-400',
        [AlertType.INFO]: 'border-blue-100 bg-background/100 dark:border-blue-400',
      },
    },
    defaultVariants: {
      variant: AlertType.INFO,
    },
  },
);

// Type for the Alert component that includes Title and Description properties
type AlertComponent = React.ForwardRefExoticComponent<
  React.HTMLAttributes<HTMLDivElement> &
  VariantProps<typeof alertVariants> &
  React.RefAttributes<HTMLDivElement>
> & {
  Title: React.ForwardRefExoticComponent<
    React.HTMLAttributes<HTMLHeadingElement> &
    React.RefAttributes<HTMLParagraphElement>
  >;
  Description: React.ForwardRefExoticComponent<
    React.HTMLAttributes<HTMLParagraphElement> &
    React.RefAttributes<HTMLParagraphElement>
  >;
};

// The Alert component
const Alert = ({ ref, className, variant, children, ...props }: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof alertVariants> & { ref?: React.RefObject<HTMLDivElement | null> }) => {
  const getIcon = () => {
    switch (variant) {
      case AlertType.ERROR:
        return <AlertCircle className="mr-3 size-5 text-red-500 dark:text-red-400" />;
      case AlertType.WARNING:
        return <AlertTriangle className="mr-3 size-5 text-yellow-600 dark:text-yellow-400" />;
      case AlertType.SUCCESS:
        return <CheckCircle className="mr-3 size-5 text-green-600 dark:text-green-400" />;
      case AlertType.INFO:
        return <Info className="mr-3 size-5 text-blue-600 dark:text-blue-400" />;
      default:
        return <Info className="mr-3 size-5 text-blue-600 dark:text-blue-400" />;
    }
  };

  return (
    <div
      ref={ref}
      role="alert"
      className={cn(alertVariants({ variant }), className)}
      {...props}
    >
      {getIcon()}
      {children}
    </div>
  );
};

Alert.displayName = 'Alert';

// The AlertTitle component
const AlertTitle = ({ ref, className, ...props }: React.HTMLAttributes<HTMLHeadingElement> & { ref?: React.RefObject<HTMLParagraphElement | null> }) => (
  <h5
    ref={ref}
    className={cn('mb-1 font-medium leading-none tracking-tight', className)}
    {...props}
  />
);

AlertTitle.displayName = 'AlertTitle';

// The AlertDescription component
const AlertDescription = ({ ref, className, ...props }: React.HTMLAttributes<HTMLParagraphElement> & { ref?: React.RefObject<HTMLParagraphElement | null> }) => (
  <div
    ref={ref}
    className={cn('text-sm [&_p]:leading-relaxed', className)}
    {...props}
  />
);

AlertDescription.displayName = 'AlertDescription';

// Attach the subcomponents to the main Alert component
Alert.Title = AlertTitle;
Alert.Description = AlertDescription;

export { Alert };
