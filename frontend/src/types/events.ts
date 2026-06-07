/**
 * SSE event contract for the Singularity simulation stream.
 *
 * These types are the boundary between the backend and the dashboard. Every
 * event the backend emits MUST conform to one of the members of `SSEvent`.
 * Keep this file authoritative: the FastAPI emitter should target this shape.
 *
 * Mirrors spec section 4.3 (SSE Event Types).
 */

export type AgentType = "web_search" | "financial" | "psychometric" | "forecast";

export type NodeStatus = "pending" | "running" | "done" | "failed";

export type CausalNodeKind = "cause" | "effect" | "mediator" | "goal";

export type InfluenceLabel = "++" | "+" | "-" | "--";

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
  /** Lexicon-derived polarity in -1..1; drives evidence-feed row coloring. */
  sentiment?: number;
}

/** A single OCEAN profile point used by radar / 3D scatter. */
export interface OceanScores {
  O: number;
  C: number;
  E: number;
  A: number;
  N: number;
}

export interface FacetScore {
  name: string;
  score: number;
  band: "high" | "moderate" | "low";
}

export interface PersonaOpinion {
  id: string;
  archetype_id: string;
  cluster: number;
  cluster_label: string;
  ocean: OceanScores;
  sentiment: number;
  behavioral_intent: string;
  emotional_state: string;
  key_concerns: string[];
  action_likelihood: number;
  /** First-person comment from cognitive deliberation. */
  comment: string;
  /** Top facet drivers (highest absolute deviation from 50). */
  top_facets: FacetScore[];
  /** Full 30-facet profile (cognitive agents). */
  facets?: Record<string, number>;
  stance_confidence?: number;
  uncertainty?: number;
  active_biases?: string[];
  response_source?: "programmatic" | "llm";
}

/** event: deliberation_ready */
export interface NarrativeCluster {
  label: string;
  size: number;
  sentiment: number;
}

export interface DeliberationPayload {
  agreement_rate: number;
  polarization_index: number;
  confidence_score: number;
  narrative_clusters: NarrativeCluster[];
  cluster_sentiments: Record<string, number>;
  cluster_actions: Record<string, number>;
  persona_archetypes: string[];
  entropy_mean: number;
  social_contagion_index: number;
}

/** event: social_interaction_tick */
export interface SocialDebate {
  cluster_a: string;
  cluster_b: string;
  topic: string;
  stance_a: number;
  stance_b: number;
  intensity: number;
}

export interface SocialPersuasionEvent {
  influencer_id: string;
  target_cluster: string;
  delta_sentiment: number;
  success_rate: number;
}

export interface SocialNarrative {
  narrative_id: string;
  label: string;
  adoption_pct: number;
  sentiment: number;
}

export interface SocialInteractionTickPayload {
  round: number;
  debates: SocialDebate[];
  persuasion_events: SocialPersuasionEvent[];
  narratives: SocialNarrative[];
  polarization_index: number;
  mean_sentiment: number;
}

/** event: social_simulation_ready */
export interface SocialSimulationPayload {
  rounds_completed: number;
  final_narratives: SocialNarrative[];
  contagion_index: number;
  polarization_index: number;
  mean_sentiment: number;
}

/** event: council_opinion / council_ready */
export type SpecialistId = "pr" | "brand" | "marketing" | "consumer_psychology";

export interface CouncilOpinionPayload {
  specialist_id: SpecialistId;
  role: string;
  recommendation: string;
  risks: string[];
  confidence: number;
}

export interface CouncilReadyPayload {
  opinions: CouncilOpinionPayload[];
  synthesis: string;
}

/** event: consensus_ready */
export interface ConsensusPayload {
  agreement_score: number;
  recommended_action: string;
  dissent: string;
  supporting_signals: string[];
  council_alignment: number;
  ranked_actions: string[];
}

/** event: persona_batch */
export interface PersonaBatchPayload {
  batch_index: number;
  batch_total: number;
  profiles_in_batch: number;
  cumulative_profiles: number;
  /** Mean OCEAN across the population so far (0..100). */
  ocean_mean: OceanScores;
  /** Down-sampled persona points for the 3D PCA scatter. */
  points: PersonaPoint[];
  /** OCEAN-facet x stimulus sentiment matrix rows for the heatmap. */
  heatmap: HeatmapRow[];
  /** Full opinion rows for this batch (typically 250). */
  opinions: PersonaOpinion[];
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
  model: string; // "TimesFM+Prophet-ICF" | "TimesFM-ICF" | "Prophet" | "Chronos" | ...
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
  root_goal: string;
  root_description: string;
  overall_prediction: number;
  nodes: CausalNode[];
  edges: CausalEdge[];
}

export interface CausalNode {
  id: string;
  label: string;
  kind: CausalNodeKind;
  prediction: number;
  criticality: number;
  description: string;
}

export interface CausalEdge {
  source: string;
  target: string;
  p_value: number; // Granger p-value
  weight: number; // Hawkes excitation / effect strength 0..1
  lag: number;
  influence: InfluenceLabel;
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
  | { type: "deliberation_ready"; payload: DeliberationPayload }
  | { type: "social_interaction_tick"; payload: SocialInteractionTickPayload }
  | { type: "social_simulation_ready"; payload: SocialSimulationPayload }
  | { type: "council_opinion"; payload: CouncilOpinionPayload }
  | { type: "council_ready"; payload: CouncilReadyPayload }
  | { type: "consensus_ready"; payload: ConsensusPayload }
  | { type: "forecast_ready"; payload: ForecastReadyPayload }
  | { type: "causal_graph"; payload: CausalGraphPayload }
  | { type: "report_section"; payload: ReportSectionPayload }
  | { type: "complete"; payload: CompletePayload }
  | { type: "error"; payload: ErrorPayload };

export type SSEventType = SSEvent["type"];

/** The named SSE event channels consumed by EventSource. */
export const SSE_EVENT_TYPES: SSEventType[] = [
  "dag_created",
  "agent_started",
  "agent_result",
  "persona_batch",
  "deliberation_ready",
  "social_interaction_tick",
  "social_simulation_ready",
  "council_opinion",
  "council_ready",
  "consensus_ready",
  "forecast_ready",
  "causal_graph",
  "report_section",
  "complete",
  "error",
];
