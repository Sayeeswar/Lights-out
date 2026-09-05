// "1987-05-11T09:30:00-04:00" -> "1987-05". Anything that doesn't start with
// YYYY-MM is returned unchanged. Used for both the x-axis ticks and the
// hover tooltip title so neither shows the time part.
function monthYear(value) {
  const match = /^(\d{4})-(\d{2})/.exec(String(value));
  return match ? `${match[1]}-${match[2]}` : String(value);
}

// Colors for a multi-series chart (e.g. Operating Cash Flow vs Free Cash
// Flow). Cycles if there are more series than colors.
const CHART_PALETTE = [
  { border: "#4f7cff", background: "rgba(79, 124, 255, 0.6)" },
  { border: "#ff8a4f", background: "rgba(255, 138, 79, 0.6)" },
  { border: "#4fd6a3", background: "rgba(79, 214, 163, 0.6)" },
  { border: "#e0538c", background: "rgba(224, 83, 140, 0.6)" },
];

function buildDatasets(chart) {
  if (Array.isArray(chart.series)) {
    return chart.series.map((s, i) => {
      const color = CHART_PALETTE[i % CHART_PALETTE.length];
      return {
        label: s.label || "",
        data: s.data,
        borderColor: color.border,
        backgroundColor: color.background,
      };
    });
  }

  return [
    {
      label: chart.label || "",
      data: chart.y,
      borderColor: "#4f7cff",
      backgroundColor: chart.chart_type === "bar"
        ? "rgba(79, 124, 255, 0.6)"
        : "rgba(79, 124, 255, 0.15)",
      fill: chart.chart_type !== "bar",
      tension: 0.25,
      pointRadius: chart.x.length > 60 ? 0 : 2,
    },
  ];
}

function renderChart(canvasId, chart) {
  const ctx = document.getElementById(canvasId);
  if (!ctx) return;

  new Chart(ctx, {
    type: chart.chart_type === "bar" ? "bar" : "line",
    data: {
      labels: chart.x,
      datasets: buildDatasets(chart),
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          ticks: {
            color: "#8a90a0",
            maxRotation: 0,
            autoSkip: true,
            maxTicksLimit: 8,
            callback(value) {
              return monthYear(this.getLabelForValue(value));
            },
          },
          grid: { color: "#262b36" },
        },
        y: {
          ticks: { color: "#8a90a0" },
          grid: { color: "#262b36" },
        },
      },
      plugins: {
        legend: { labels: { color: "#e6e8ec" } },
        tooltip: {
          callbacks: {
            title(items) {
              return items.length ? monthYear(items[0].label) : "";
            },
          },
        },
      },
    },
  });
}
