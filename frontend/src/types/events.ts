/**
 * SSE event contract for the Singularity simulation stream.
 *
 * These types are the boundary between the backend (mock today, CrewAI Flow
 * later) and the dashboard. Every event the backend emits MUST conform to one
 * of the members of `SSEvent`. Keep this file authoritative: the mock server
 * scenario and the real FastAPI emitter should both target this shape.
 *
 * Mirrors spec section 4.3 (SSE Event Types).
 */

export type AgentType = "web_search" | "financial" | "psychometric" | "forecast";

export type NodeStatus = "pending" | "running" | "done" | "failed";

export interface DagNode {
  id: string;
  task: string;
  agent_type: AgentType;
  dependencies: string[];
  priority: number;
}

export interface DagEdge {
  source: string;
  target: string;
}

/** event: dag_created */
export interface DagCreatedPayload {
  root_query: string;
  nodes: DagNode[];
  edges: DagEdge[];
}

/** event: agent_started */
export interface AgentStartedPayload {
  agent_id: string;
  task: string;
  agent_type: AgentType;
}

/** event: agent_result */
export interface AgentResultPayload {
  agent_id: string;
  agent_type: AgentType;
  /** Free-form evidence snippets for the evidence feed. */
  data: EvidenceItem[];
  confidence: number; // 0..1
  duration_ms: number;
}

export interface EvidenceItem {
  source: string; // e.g. "yFinance", "DuckDuckGo"
  title: string;
  detail: string;
  value?: number;
  unit?: string;
  url?: string;
}

/** A single OCEAN profile point used by radar / 3D scatter. */
export interface OceanScores {
  O: number;
  C: number;
  E: number;
  A: number;
  N: number;
}

/** event: persona_batch */
export interface PersonaBatchPayload {
  batch_index: number;
  batch_total: number;
  profiles_in_batch: number;
  cumulative_profiles: number;
  /** Mean OCEAN across the population so far (0..100). */
  ocean_mean: OceanScores;
  /** Sentiment histogram buckets from -1..1. */
  sentiment_dist: { bucket: number; count: number }[];
  /** Down-sampled persona points for the 3D PCA scatter. */
  points: PersonaPoint[];
  /** OCEAN-facet x stimulus sentiment matrix rows for the heatmap. */
  heatmap: HeatmapRow[];
}

export interface PersonaPoint {
  id: string;
  /** PCA-reduced coordinates of the 300-item IPIP vector. */
  pca: [number, number, number];
  ocean: OceanScores;
  sentiment: number; // -1..1
  cluster: number;
}

export interface HeatmapRow {
  facet: string;
  values: number[]; // sentiment per stimulus column, -1..1
}

/** event: forecast_ready */
export interface ForecastReadyPayload {
  model: string; // "TimesFM-ICF" | "Chronos" | ...
  metric: string;
  horizon_days: number;
  history: ForecastPoint[];
  predictions: ForecastPoint[];
  intervals: { date: string; lower: number; upper: number }[];
  mase_score: number;
}

export interface ForecastPoint {
  date: string; // ISO date
  value: number;
}

/** event: causal_graph */
export interface CausalGraphPayload {
  nodes: CausalNode[];
  edges: CausalEdge[];
}

export interface CausalNode {
  id: string;
  label: string;
  kind: "cause" | "effect" | "mediator";
}

export interface CausalEdge {
  source: string;
  target: string;
  p_value: number; // Granger p-value
  weight: number; // Hawkes excitation / effect strength 0..1
  lag: number;
}

/** event: report_section */
export interface ReportSectionPayload {
  index: number;
  total: number;
  section: string; // section title
  content: string; // markdown
  status: "streaming" | "final";
}

/** event: complete */
export interface CompletePayload {
  session_id: string;
  duration_ms: number;
  nodes_resolved: number;
}

/** event: error */
export interface ErrorPayload {
  code: string;
  message: string;
  node_id?: string;
}

/** Discriminated union of every event the stream can deliver. */
export type SSEvent =
  | { type: "dag_created"; payload: DagCreatedPayload }
  | { type: "agent_started"; payload: AgentStartedPayload }
  | { type: "agent_result"; payload: AgentResultPayload }
  | { type: "persona_batch"; payload: PersonaBatchPayload }
  | { type: "forecast_ready"; payload: ForecastReadyPayload }
  | { type: "causal_graph"; payload: CausalGraphPayload }
  | { type: "report_section"; payload: ReportSectionPayload }
  | { type: "complete"; payload: CompletePayload }
  | { type: "error"; payload: ErrorPayload };

export type SSEventType = SSEvent["type"];

/** The named SSE event channels (used by both EventSource and the mock). */
export const SSE_EVENT_TYPES: SSEventType[] = [
  "dag_created",
  "agent_started",
  "agent_result",
  "persona_batch",
  "forecast_ready",
  "causal_graph",
  "report_section",
  "complete",
  "error",
];
