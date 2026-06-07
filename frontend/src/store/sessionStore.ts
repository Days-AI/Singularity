import { create } from "zustand";
import type {
  AgentResultPayload,
  CausalGraphPayload,
  CompletePayload,
  DagCreatedPayload,
  DagNode,
  ErrorPayload,
  EvidenceItem,
  ForecastReadyPayload,
  NodeStatus,
  OceanScores,
  PersonaBatchPayload,
  PersonaPoint,
  HeatmapRow,
  ReportSectionPayload,
  SSEvent,
} from "@/types/events";

export type ConnectionStatus =
  | "idle"
  | "connecting"
  | "streaming"
  | "complete"
  | "error";

export interface ReportSection {
  index: number;
  title: string;
  content: string;
  status: "streaming" | "final";
}

export interface EvidenceEntry extends EvidenceItem {
  agentId: string;
  agentType: string;
  confidence: number;
  ts: number;
}

export interface Toast {
  id: string;
  kind: "error" | "info";
  message: string;
}

interface SessionState {
  connection: ConnectionStatus;
  sessionId: string | null;
  rootQuery: string;
  startedAt: number | null;
  durationMs: number | null;

  // DAG
  dagNodes: DagNode[];
  dagEdges: DagCreatedPayload["edges"];
  nodeStatus: Record<string, NodeStatus>;

  // Evidence feed (newest first, capped)
  evidence: EvidenceEntry[];

  // Psychometrics
  oceanMean: OceanScores | null;
  sentimentDist: { bucket: number; count: number }[];
  personaPoints: PersonaPoint[];
  heatmap: HeatmapRow[];
  personasSimulated: number;
  personaTarget: number;

  // Forecast
  forecast: ForecastReadyPayload | null;

  // Causal
  causal: CausalGraphPayload | null;

  // Report
  reportSections: ReportSection[];

  // Derived metrics
  activeAgents: number;
  toasts: Toast[];

  // actions
  apply: (event: SSEvent) => void;
  setConnection: (status: ConnectionStatus) => void;
  setSessionMeta: (meta: { sessionId?: string | null; rootQuery?: string }) => void;
  setReportSections: (sections: ReportSectionPayload[]) => void;
  pushToast: (toast: Omit<Toast, "id">) => void;
  reset: () => void;
  dismissToast: (id: string) => void;
}

const EVIDENCE_CAP = 200;
const PERSONA_POINT_CAP = 1500;

const initialState = {
  connection: "idle" as ConnectionStatus,
  sessionId: null,
  rootQuery: "",
  startedAt: null,
  durationMs: null,
  dagNodes: [],
  dagEdges: [],
  nodeStatus: {},
  evidence: [],
  oceanMean: null,
  sentimentDist: [],
  personaPoints: [],
  heatmap: [],
  personasSimulated: 0,
  personaTarget: 1500,
  forecast: null,
  causal: null,
  reportSections: [],
  activeAgents: 0,
};

export const useSessionStore = create<SessionState>((set) => ({
  ...initialState,
  toasts: [],

  setConnection: (status) =>
    set((s) => ({
      connection: status,
      startedAt:
        status === "connecting" || status === "streaming"
          ? s.startedAt ?? Date.now()
          : s.startedAt,
    })),

  setSessionMeta: (meta) =>
    set((s) => ({
      sessionId: meta.sessionId !== undefined ? meta.sessionId : s.sessionId,
      rootQuery: meta.rootQuery !== undefined ? meta.rootQuery : s.rootQuery,
    })),

  setReportSections: (sections) =>
    set({
      reportSections: sections
        .map((p) => ({
          index: p.index,
          title: p.section,
          content: p.content,
          status: p.status,
        }))
        .sort((a, b) => a.index - b.index),
    }),

  pushToast: (toast) =>
    set((s) => ({
      toasts: [...s.toasts, { ...toast, id: `toast-${Date.now()}` }],
    })),

  reset: () => set({ ...initialState, toasts: [] }),

  dismissToast: (id) =>
    set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),

  apply: (event) =>
    set((state) => {
      switch (event.type) {
        case "dag_created":
          return applyDagCreated(state, event.payload);
        case "agent_started":
          return applyAgentStarted(state, event.payload.agent_id);
        case "agent_result":
          return applyAgentResult(state, event.payload);
        case "persona_batch":
          return applyPersonaBatch(state, event.payload);
        case "forecast_ready":
          return { forecast: event.payload };
        case "causal_graph":
          return { causal: event.payload };
        case "report_section":
          return applyReportSection(state, event.payload);
        case "complete":
          return applyComplete(event.payload);
        case "error":
          return applyError(state, event.payload);
        default: {
          // Exhaustiveness guard: if a new event type is added to SSEvent and
          // not handled here, TypeScript will flag this assignment.
          const _exhaustive: never = event;
          return _exhaustive;
        }
      }
    }),
}));

function applyDagCreated(
  state: SessionState,
  p: DagCreatedPayload
): Partial<SessionState> {
  const nodeStatus: Record<string, NodeStatus> = {};
  for (const n of p.nodes) nodeStatus[n.id] = "pending";
  return {
    rootQuery: p.root_query,
    dagNodes: p.nodes,
    dagEdges: p.edges,
    nodeStatus,
    connection: state.connection === "complete" ? state.connection : "streaming",
  };
}

function applyAgentStarted(
  state: SessionState,
  agentId: string
): Partial<SessionState> {
  return {
    nodeStatus: { ...state.nodeStatus, [agentId]: "running" },
    activeAgents: state.activeAgents + 1,
  };
}

function applyAgentResult(
  state: SessionState,
  p: AgentResultPayload
): Partial<SessionState> {
  const newEntries: EvidenceEntry[] = p.data.map((item) => ({
    ...item,
    agentId: p.agent_id,
    agentType: p.agent_type,
    confidence: p.confidence,
    ts: Date.now(),
  }));
  return {
    nodeStatus: { ...state.nodeStatus, [p.agent_id]: "done" },
    activeAgents: Math.max(0, state.activeAgents - 1),
    evidence: [...newEntries, ...state.evidence].slice(0, EVIDENCE_CAP),
  };
}

function applyPersonaBatch(
  state: SessionState,
  p: PersonaBatchPayload
): Partial<SessionState> {
  return {
    oceanMean: p.ocean_mean,
    sentimentDist: p.sentiment_dist,
    heatmap: p.heatmap,
    personasSimulated: p.cumulative_profiles,
    personaPoints: [...state.personaPoints, ...p.points].slice(
      0,
      PERSONA_POINT_CAP
    ),
  };
}

function applyReportSection(
  state: SessionState,
  p: ReportSectionPayload
): Partial<SessionState> {
  const existingIdx = state.reportSections.findIndex((s) => s.index === p.index);
  const next: ReportSection = {
    index: p.index,
    title: p.section,
    content: p.content,
    status: p.status,
  };
  const sections = [...state.reportSections];
  if (existingIdx >= 0) sections[existingIdx] = next;
  else sections.push(next);
  sections.sort((a, b) => a.index - b.index);
  return { reportSections: sections };
}

function applyComplete(p: CompletePayload): Partial<SessionState> {
  return {
    connection: "complete",
    sessionId: p.session_id,
    durationMs: p.duration_ms,
    activeAgents: 0,
  };
}

function applyError(
  state: SessionState,
  p: ErrorPayload
): Partial<SessionState> {
  const toast: Toast = {
    id: `err-${Date.now()}`,
    kind: "error",
    message: p.node_id ? `[${p.node_id}] ${p.message}` : p.message,
  };
  const nodeStatus = p.node_id
    ? { ...state.nodeStatus, [p.node_id]: "failed" as NodeStatus }
    : state.nodeStatus;
  return {
    connection: "error",
    nodeStatus,
    toasts: [...state.toasts, toast],
  };
}
