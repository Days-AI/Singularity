import { memo, type ReactNode } from "react";
import { Panel, type PanelStatus } from "./Panel";
import {
  useSessionStore,
  type ConnectionStatus,
} from "@/store/sessionStore";

function panelStatus(
  connection: ConnectionStatus,
  hasData: boolean
): PanelStatus {
  if (connection === "error") return hasData ? "done" : "error";
  if (hasData) return connection === "complete" ? "done" : "live";
  return connection === "streaming" || connection === "connecting"
    ? "live"
    : "idle";
}

interface DashboardPanelProps {
  label: string;
  code?: string;
  order?: number;
  flush?: boolean;
  className?: string;
  hasData: boolean;
  children: ReactNode;
}

/** Panel shell with an isolated status subscription so sibling panels do not re-render. */
const DashboardPanelInner = ({
  label,
  code,
  order,
  flush,
  className,
  hasData,
  children,
}: DashboardPanelProps) => {
  const connection = useSessionStore((s) => s.connection);
  return (
    <Panel
      label={label}
      code={code}
      status={panelStatus(connection, hasData)}
      order={order}
      flush={flush}
      className={className}
    >
      {children}
    </Panel>
  );
};

export const DashboardPanel = memo(DashboardPanelInner);
