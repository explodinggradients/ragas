import { useEffect, useState, useMemo } from 'react';
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from '@tanstack/react-table';
import { TitleHeader } from '@/components/TitleHeader';
import { backendApiClient } from '@/helpers/api/backend-api-client';
import { backendApiRoutes } from '@/helpers/api/api-routes';
import { useAlerts } from '@/components/Alerts/context/AlertsContext';
import { DataTableShimmer } from '@/components/ui/data-table-shimmer';

interface DatasetFile {
  filename: string;
  created_at: number;
  modified_at: number;
  size: number;
}

const columnHelper = createColumnHelper<DatasetFile>();

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetFile[]>([]);
  const [loading, setLoading] = useState(true);
  const [sorting, setSorting] = useState<SortingState>([]);
  const { showErrorAlert } = useAlerts();

  const columns = useMemo(
    () => [
      columnHelper.accessor('filename', {
        header: 'Filename',
        cell: (info) => (
          <div className="font-medium">{info.getValue()}</div>
        ),
      }),
      columnHelper.accessor('modified_at', {
        header: 'Modified',
        cell: (info) => (
          <div className="text-sm text-muted-foreground">
            {new Date(info.getValue() * 1000).toLocaleDateString()}
          </div>
        ),
      }),
      columnHelper.accessor('size', {
        header: 'Size',
        cell: (info) => (
          <div className="text-sm text-muted-foreground">
            {formatFileSize(info.getValue())}
          </div>
        ),
      }),
    ],
    []
  );

  const table = useReactTable({
    data: datasets,
    columns,
    state: {
      sorting,
    },
    onSortingChange: setSorting,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
  });

  useEffect(() => {
    const fetchDatasets = async () => {
      try {
        setLoading(true);
        const data = await backendApiClient.get<DatasetFile[]>(backendApiRoutes.datasets.list());
        setDatasets(data);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to fetch datasets';
        showErrorAlert(errorMessage, 'Error loading datasets');
      } finally {
        setLoading(false);
      }
    };

    fetchDatasets();
  }, [showErrorAlert]);

  function formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }

  return (
    <div className="flex flex-1 flex-col gap-4 p-4 pt-0">
      <TitleHeader title="Datasets" />
      
      <div className="rounded-md border">
        {loading ? (
          <div className="p-4">
            <DataTableShimmer columns={3} rows={5} />
          </div>
        ) : (
          <>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  {table.getHeaderGroups().map((headerGroup) => (
                    <tr key={headerGroup.id} className="border-b">
                      {headerGroup.headers.map((header) => (
                        <th
                          key={header.id}
                          className="px-4 py-3 text-left text-sm font-medium text-muted-foreground cursor-pointer hover:bg-muted/50"
                          onClick={header.column.getToggleSortingHandler()}
                        >
                          {header.isPlaceholder
                            ? null
                            : flexRender(header.column.columnDef.header, header.getContext())}
                          {header.column.getIsSorted() === 'asc' && ' ↑'}
                          {header.column.getIsSorted() === 'desc' && ' ↓'}
                        </th>
                      ))}
                    </tr>
                  ))}
                </thead>
                <tbody>
                  {table.getRowModel().rows.map((row) => (
                    <tr key={row.id} className="border-b hover:bg-muted/50">
                      {row.getVisibleCells().map((cell) => (
                        <td key={cell.id} className="px-4 py-3">
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
            {datasets.length === 0 && (
              <div className="p-8 text-center text-muted-foreground">
                No datasets found
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
