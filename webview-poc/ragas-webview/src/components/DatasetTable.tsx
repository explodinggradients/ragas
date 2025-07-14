import { CheckCircle, XCircle } from 'lucide-react';

interface DatasetRow {
  query: string;
  response: string;
  correctness: 'pass' | 'fail';
}

interface DatasetTableProps {
  data: DatasetRow[];
  totalRows: number;
  filename: string;
}

export function DatasetTable({ data, totalRows, filename }: DatasetTableProps) {
  const passCount = data.filter(row => row.correctness === 'pass').length;
  const failCount = data.filter(row => row.correctness === 'fail').length;

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900 mb-2">Dataset Viewer</h1>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4 text-sm text-gray-600">
            <span>File: <span className="font-medium">{filename}</span></span>
            <span>Total Rows: <span className="font-medium">{totalRows}</span></span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-green-600">
              <CheckCircle className="w-4 h-4" />
              <span className="font-medium">Pass: {passCount}</span>
            </div>
            <div className="flex items-center gap-2 text-red-600">
              <XCircle className="w-4 h-4" />
              <span className="font-medium">Fail: {failCount}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Simple Table */}
      <div className="overflow-hidden rounded-lg border border-gray-200">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Query
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Response
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Correctness
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((row, index) => (
              <tr key={index} className="hover:bg-gray-50">
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  <div className="max-w-md truncate" title={row.query}>
                    {row.query}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  <div className="max-w-md truncate" title={row.response}>
                    {row.response}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                  <div className={`flex items-center gap-2 ${row.correctness === 'pass' ? 'text-green-600' : 'text-red-600'}`}>
                    {row.correctness === 'pass' ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                    <span className="font-medium capitalize">{row.correctness}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}