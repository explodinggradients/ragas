import { useEffect, useState } from 'react';
import { API_ENDPOINTS } from './config';
import { DatasetTable } from './components/DatasetTable';
import { LoadingSpinner } from './components/LoadingSpinner';
import { ErrorMessage } from './components/ErrorMessage';

interface DatasetRow {
  query: string;
  response: string;
  correctness: 'pass' | 'fail';
}

interface DatasetResponse {
  filename: string;
  total_rows: number;
  columns: string[];
  data: DatasetRow[];
}

function App() {
  const [dataset, setDataset] = useState<DatasetResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchDataset = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch(API_ENDPOINTS.DATASETS);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch dataset: ${response.status} ${response.statusText}`);
      }
      
      const data: DatasetResponse = await response.json();
      setDataset(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDataset();
  }, []);

  if (loading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <ErrorMessage message={error} onRetry={fetchDataset} />;
  }

  if (!dataset) {
    return <ErrorMessage message="No dataset found" onRetry={fetchDataset} />;
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <DatasetTable
          data={dataset.data}
          totalRows={dataset.total_rows}
          filename={dataset.filename}
        />
      </div>
    </div>
  );
}

export default App;