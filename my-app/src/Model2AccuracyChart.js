import { PieChart, Pie, Legend, Cell } from 'recharts';

const Model2AccuracyChart = ({ data }) => {
  const COLORS = ['#107C10', '#D80000'];

  return (
    <PieChart width={500} height={500}>
      <Pie
        data={data}
        dataKey="value"
        cx="50%"
        cy="50%"
        outerRadius={160}
        fill="transparent"
        stroke="none"
        labelLine={false}
        label={({ cx, cy, midAngle, innerRadius, outerRadius, percent }) => {
          const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
          const x = cx + radius * Math.cos(-midAngle * (Math.PI / 180));
          const y = cy + radius * Math.sin(-midAngle * (Math.PI / 180));

          if (percent > 0) {
            return (
              <text
                x={x}
                y={y}
                fill="#fff"
                textAnchor={x > cx ? 'start' : 'end'}
                dominantBaseline="central"
              >
                {`${(percent * 100).toFixed(0)}%`}
              </text>
            );
          } else {
            return null;
          }
        }}
      >
        {data.map((entry, index) => (
          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
        ))}
      </Pie>
      <Legend
        iconType="circle"
        verticalAlign="bottom"
        align="center"
        layout="vertical"
        wrapperStyle={{ marginTop: '100px' }}
      />
    </PieChart>
  );
};

export default Model2AccuracyChart;
