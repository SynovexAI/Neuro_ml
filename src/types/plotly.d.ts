// Minimal typing for the pre-bundled Plotly dist. The real Plotly types are huge
// and we only touch react/newPlot/purge/relayout, so keep this lightweight.
declare module "plotly.js-dist-min" {
  export type PlotlyData = Record<string, unknown>;
  export type PlotlyLayout = Record<string, unknown>;
  export type PlotlyConfig = Record<string, unknown>;
  export function react(root: HTMLElement, data: PlotlyData[], layout?: PlotlyLayout, config?: PlotlyConfig): Promise<HTMLElement>;
  export function newPlot(root: HTMLElement, data: PlotlyData[], layout?: PlotlyLayout, config?: PlotlyConfig): Promise<HTMLElement>;
  export function relayout(root: HTMLElement, layout: PlotlyLayout): Promise<HTMLElement>;
  export function purge(root: HTMLElement): void;
  const Plotly: {
    react: typeof react;
    newPlot: typeof newPlot;
    relayout: typeof relayout;
    purge: typeof purge;
  };
  export default Plotly;
}
