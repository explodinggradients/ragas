import { cn } from '@/lib/utils';

interface DataTableShimmerProps {
  columns?: number;
  rows?: number;
  className?: string;
}

export function DataTableShimmer({ 
  columns = 3, 
  rows = 5, 
  className 
}: DataTableShimmerProps) {
  return (
    <div className={cn("w-full", className)}>
      {/* Table Header Shimmer */}
      <div className="flex space-x-4 mb-4">
        {Array.from({ length: columns }).map((_, index) => (
          <div
            key={`header-${index}`}
            className="h-4 bg-muted animate-pulse rounded flex-1"
          />
        ))}
      </div>
      
      {/* Table Rows Shimmer */}
      <div className="space-y-3">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <div key={`row-${rowIndex}`} className="flex space-x-4">
            {Array.from({ length: columns }).map((_, colIndex) => (
              <div
                key={`cell-${rowIndex}-${colIndex}`}
                className={cn(
                  "h-6 bg-muted animate-pulse rounded",
                  colIndex === 0 ? "flex-2" : "flex-1" // Make first column wider
                )}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

interface CardGridShimmerProps {
  items?: number;
  className?: string;
}

export function CardGridShimmer({ items = 6, className }: CardGridShimmerProps) {
  return (
    <div className={cn("grid auto-rows-min gap-4 md:grid-cols-3", className)}>
      {Array.from({ length: items }).map((_, index) => (
        <div
          key={`card-${index}`}
          className="aspect-video rounded-xl bg-muted/50 animate-pulse p-4"
        >
          <div className="h-4 bg-muted animate-pulse rounded w-3/4 mb-2" />
          <div className="h-3 bg-muted animate-pulse rounded w-1/2" />
        </div>
      ))}
    </div>
  );
}