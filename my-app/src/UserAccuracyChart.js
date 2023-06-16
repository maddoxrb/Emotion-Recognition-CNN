import { PieChart, Pie, Legend, Cell } from 'recharts';

const UserAccuracyChart = ({ data, cardWidth }) => {
  const COLORS = ['#107C10', '#D80000'];

  const chartWidth = 0.4; // Percentage value for chart width relative to the card

  const chartRadius = cardWidth * chartWidth * 0.55;

  return (
    <PieChart width={cardWidth} height={cardWidth}>
      <Pie
        data={data}
        dataKey="value"
        cx="50%"
        cy="50%"
        outerRadius={chartRadius}
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
    </PieChart>
  );
};

export default UserAccuracyChart;
