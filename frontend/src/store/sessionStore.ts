import { create } from "zustand";
import { logSseEvent } from "@/lib/masterLog";
import type {
  AgentResultPayload,
  CausalGraphPayload,
  CompletePayload,
  ConsensusPayload,
  CouncilOpinionPayload,
  CouncilReadyPayload,
  DagCreatedPayload,
  DagNode,
  ErrorPayload,
  EvidenceItem,
  DeliberationPayload,
  ForecastReadyPayload,
  NodeStatus,
  OceanScores,
  PersonaBatchPayload,
  PersonaOpinion,
  PersonaPoint,
  HeatmapRow,
  ReportSectionPayload,
  SocialInteractionTickPayload,
  SocialSimulationPayload,
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

  dagNodes: DagNode[];
  dagEdges: DagCreatedPayload["edges"];
  nodeStatus: Record<string, NodeStatus>;

  evidence: EvidenceEntry[];

  oceanMean: OceanScores | null;
  personaPoints: PersonaPoint[];
  personaOpinions: PersonaOpinion[];
  heatmap: HeatmapRow[];
  personasSimulated: number;
  personaTarget: number;

  deliberation: DeliberationPayload | null;

  socialTicks: SocialInteractionTickPayload[];
  socialSimulation: SocialSimulationPayload | null;
  councilOpinions: CouncilOpinionPayload[];
  council: CouncilReadyPayload | null;
  consensus: ConsensusPayload | null;

  forecast: ForecastReadyPayload | null;
  causal: CausalGraphPayload | null;
  reportSections: ReportSection[];

  activeAgents: number;
  toasts: Toast[];

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
const PERSONA_OPINION_CAP = 1500;
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
  personaPoints: [],
  personaOpinions: [],
  heatmap: [],
  personasSimulated: 0,
  personaTarget: 1500,
  deliberation: null,
  socialTicks: [],
  socialSimulation: null,
  councilOpinions: [],
  council: null,
  consensus: null,
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

  apply: (event) => {
    const sessionId = useSessionStore.getState().sessionId;
    if (
      event.type === "persona_batch" ||
      event.type === "agent_result" ||
      event.type === "complete" ||
      event.type === "error"
    ) {
      logSseEvent(
        event.type,
        event.payload as unknown as Record<string, unknown>,
        sessionId
      );
    }
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
        case "deliberation_ready":
          return { deliberation: event.payload };
        case "social_interaction_tick":
          return { socialTicks: [...state.socialTicks, event.payload] };
        case "social_simulation_ready":
          return { socialSimulation: event.payload };
        case "council_opinion":
          return {
            councilOpinions: [...state.councilOpinions, event.payload],
          };
        case "council_ready":
          return {
            council: event.payload,
            councilOpinions: event.payload.opinions,
          };
        case "consensus_ready":
          return { consensus: event.payload };
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
          const _exhaustive: never = event;
          return _exhaustive;
        }
      }
    });
  },
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
    heatmap: p.heatmap,
    personasSimulated: p.cumulative_profiles,
    personaTarget: Math.max(state.personaTarget, p.cumulative_profiles),
    personaPoints: [...state.personaPoints, ...p.points].slice(0, PERSONA_POINT_CAP),
    personaOpinions: [...state.personaOpinions, ...(p.opinions ?? [])].slice(
      0,
      PERSONA_OPINION_CAP
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
