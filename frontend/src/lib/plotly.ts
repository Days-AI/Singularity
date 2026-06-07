// Bundle the slim Plotly distribution and wire it into react-plotly.js via the
// factory. This keeps the bundle smaller than the full plotly.js build while
// still supporting scatter/scatter3d/scatterpolar/heatmap traces we need.
import Plotly from "plotly.js-dist-min";
import createPlotlyComponent from "react-plotly.js/factory";

const Plot = createPlotlyComponent(Plotly);

export default Plot;
