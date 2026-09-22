import type {
  CausalGraphPayload,
  EvidenceItem,
  ForecastReadyPayload,
} from "@/types/events";

export type IntelNodeType = "goal" | "forecast" | "event" | "risk" | "market" | "driver";
export type EdgePolarity = "positive" | "negative";

export interface IntelNode {
  id: string;
  label: string;
  node_type: IntelNodeType;
  question: string;
  description: string;
  probability: number;
  confidence: number;
  criticality: number;
  layer: number;
  group_id: string | null;
  evidence_ids: number[];
  forecast_attached: boolean;
  trend: number[];
}

export interface IntelEdge {
  id: string;
  source: string;
  target: string;
  polarity: EdgePolarity;
  weight: number;
  confidence: number;
  lag: number;
  flow_rate: number;
  label: string;
}

export interface FlowNodeData {
  label: string;
  question: string;
  probability: number;
  confidence: number;
  criticality: number;
  nodeType: IntelNodeType;
  trend: number[];
  groupId?: string | null;
}

export interface FlowNodeDTO {
  id: string;
  type: string;
  position: { x: number; y: number };
  data: FlowNodeData;
}

export interface FlowEdgeDTO {
  id: string;
  source: string;
  target: string;
  type: string;
  data: {
    polarity: EdgePolarity;
    weight: number;
    label: string;
    flowRate: number;
  };
}

export interface CausalIntelGraph {
  session_id: string | null;
  flow_uuid: string | null;
  query: string;
  root_goal: string;
  root_description: string;
  overall_prediction: number;
  nodes: IntelNode[];
  edges: IntelEdge[];
  flow_nodes: FlowNodeDTO[];
  flow_edges: FlowEdgeDTO[];
  monte_carlo: Record<string, unknown> | null;
  prediction_market: Record<string, unknown> | null;
}

export interface GraphSnapshot {
  query: string;
  causal: CausalGraphPayload | null;
  evidence: EvidenceItem[];
  forecast: ForecastReadyPayload | null;
  metrics: Record<string, unknown>;
  deliberation: Record<string, unknown> | null;
  consensus: Record<string, unknown> | null;
}

export interface ScenarioAssumption {
  node_id: string;
  enabled: boolean;
  probability_override?: number | null;
}

export interface ScenarioResult {
  goal_probability: number;
  baseline_goal: number;
  delta: number;
  node_deltas: Record<string, number>;
  posterior_by_node: Record<string, number>;
}

export interface LinkedEvidence {
  index: number;
  source: string;
  title: string;
  detail: string;
  sentiment?: number | null;
  url?: string | null;
}

export interface EdgeSummary {
  id: string;
  source_id: string;
  source_label: string;
  target_id: string;
  target_label: string;
  polarity: EdgePolarity;
  weight: number;
  label: string;
}

export interface NodeDetail {
  node: IntelNode;
  incoming: EdgeSummary[];
  outgoing: EdgeSummary[];
  evidence: LinkedEvidence[];
  forecast: { trend: number[]; points: number } | null;
  monte_carlo: Record<string, unknown> | null;
  paths_to_goal: string[][];
  explanation: string | null;
}

export interface InfluenceRank {
  node_id: string;
  label: string;
  score: number;
  polarity: EdgePolarity | null;
}

export interface SensitivityBar {
  node_id: string;
  label: string;
  low: number;
  high: number;
  baseline: number;
}

export interface TimelineEvent {
  date: string;
  label: string;
  kind: string;
  value: number | null;
}

export interface AnalyticsBundle {
  influence_ranking: InfluenceRank[];
  sensitivity: SensitivityBar[];
  timeline: TimelineEvent[];
  monte_carlo: Record<string, unknown> | null;
}

export interface HMMState {
  name: string;
  probability: number;
}

export interface HMMResult {
  states: HMMState[];
  current_state: string;
  transition_matrix: number[][];
}

export interface PathDiscovery {
  paths: string[][];
  labels: string[][];
}
