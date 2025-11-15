import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';

interface DonutChartData {
  name: string;
  value: number;
  color: string;
  [key: string]: any;
}

interface DonutChartProps {
  data: DonutChartData[];
  centerLabel?: string;
  centerValue?: string | number;
}

const DonutChart: React.FC<DonutChartProps> = ({ data, centerLabel, centerValue }) => {
  const total = data.reduce((sum, item) => sum + item.value, 0);

  // Custom label for center
  const renderCenterLabel = () => {
    if (!centerLabel && !centerValue) return null;
    
    return (
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        {centerValue && (
          <div className="text-4xl font-bold text-white">{centerValue}</div>
        )}
        {centerLabel && (
          <div className="text-sm text-gray-400 mt-1">{centerLabel}</div>
        )}
      </div>
    );
  };

  return (
    <div className="relative w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={90}
            paddingAngle={2}
            dataKey="value"
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Pie>
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const data = payload[0];
                const percentage = total > 0 ? ((data.value as number / total) * 100).toFixed(1) : 0;
                return (
                  <div className="bg-background-secondary border border-gray-700 rounded-lg p-3 shadow-xl">
                    <p className="text-white font-medium">{data.name}</p>
                    <p className="text-gray-400 text-sm">{data.value} models ({percentage}%)</p>
                  </div>
                );
              }
              return null;
            }}
          />
          <Legend
            verticalAlign="bottom"
            height={36}
            content={({ payload }) => (
              <div className="flex justify-center gap-6 mt-4">
                {payload?.map((entry, index) => (
                  <div key={`legend-${index}`} className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: entry.color }}
                    />
                    <span className="text-sm text-gray-400">
                      {entry.value} ({entry.payload?.value || 0})
                    </span>
                  </div>
                ))}
              </div>
            )}
          />
        </PieChart>
      </ResponsiveContainer>
      {renderCenterLabel()}
    </div>
  );
};

export default DonutChart;