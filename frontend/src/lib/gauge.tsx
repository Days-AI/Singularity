import { COLORS, FONT_MONO } from "@/lib/theme";

export function outcomeColor(pct: number): string {
  if (pct >= 65) return COLORS.positive;
  if (pct >= 40) return COLORS.orange;
  return COLORS.alert;
}

export function outcomeBand(pct: number): string {
  if (pct >= 65) return "Strong";
  if (pct >= 40) return "Mixed";
  return "Weak";
}

export function gaugePoint(cx: number, cy: number, r: number, deg: number) {
  const rad = (deg * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy + r * Math.sin(rad) };
}

export function topArcPath(
  cx: number,
  cy: number,
  r: number,
  startDeg: number,
  endDeg: number
): string {
  const s = gaugePoint(cx, cy, r, startDeg);
  const e = gaugePoint(cx, cy, r, endDeg);
  let delta = endDeg - startDeg;
  if (delta < 0) delta += 360;
  const largeArc = delta > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${largeArc} 1 ${e.x} ${e.y}`;
}

/** Shared semicircle outcome gauge (0% left → 100% right). */
export function OutcomeGaugeSvg({
  value,
  label = "OUTCOME",
  className = "",
  decimals = 1,
}: {
  value: number;
  label?: string;
  className?: string;
  decimals?: number;
}) {
  const pct = Math.max(0, Math.min(100, value));
  const color = outcomeColor(pct);
  const display = decimals <= 0 ? pct.toFixed(0) : pct.toFixed(decimals);

  // Pivot at bottom-center; arc sweeps over the top (classic speedometer).
  const cx = 60;
  const cy = 54;
  const r = 38;
  const sw = 5.5;
  const valueEndDeg = 180 + (pct / 100) * 180;
  const ticks = [0, 25, 50, 75, 100];
  const leftTick = gaugePoint(cx, cy, r + 10, 180);
  const rightTick = gaugePoint(cx, cy, r + 10, 360);

  return (
    <div className={`grid h-full w-full place-items-center ${className}`}>
      <svg
        viewBox="0 0 120 72"
        className="block h-full w-full max-h-full max-w-full"
        preserveAspectRatio="xMidYMid meet"
        aria-label={`${label} ${display} percent, ${outcomeBand(pct)}`}
        role="img"
      >
        <path
          d={topArcPath(cx, cy, r, 180, 360)}
          fill="none"
          stroke={COLORS.grid}
          strokeWidth={sw}
          strokeLinecap="round"
        />
        {pct > 0 && (
          <path
            d={topArcPath(cx, cy, r, 180, valueEndDeg)}
            fill="none"
            stroke={color}
            strokeWidth={sw}
            strokeLinecap="round"
          />
        )}
        {ticks.map((t) => {
          const deg = 180 + (t / 100) * 180;
          const outer = gaugePoint(cx, cy, r + 1, deg);
          const inner = gaugePoint(cx, cy, r - (t % 50 === 0 ? 5 : 2.5), deg);
          return (
            <line
              key={t}
              x1={inner.x}
              y1={inner.y}
              x2={outer.x}
              y2={outer.y}
              stroke={COLORS.muted}
              strokeWidth={t % 50 === 0 ? 1.1 : 0.7}
              opacity={0.65}
            />
          );
        })}
        <text
          x={leftTick.x}
          y={leftTick.y + 2}
          textAnchor="middle"
          fill={COLORS.muted}
          fontFamily={FONT_MONO}
          fontSize={6}
        >
          0
        </text>
        <text
          x={rightTick.x}
          y={rightTick.y + 2}
          textAnchor="middle"
          fill={COLORS.muted}
          fontFamily={FONT_MONO}
          fontSize={6}
        >
          100
        </text>
        <text
          x={cx}
          y={cy - 14}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontFamily={FONT_MONO}
          fontSize={20}
          fontWeight={700}
        >
          {display}
          <tspan fontSize={11} fontWeight={600}>
            %
          </tspan>
        </text>
        <text
          x={cx}
          y={cy - 2}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={COLORS.muted}
          fontFamily={FONT_MONO}
          fontSize={6}
          letterSpacing="0.1em"
        >
          {label}
        </text>
        <rect x={cx - 20} y={58} width={40} height={10} rx={2} fill={`${color}22`} />
        <text
          x={cx}
          y={64}
          textAnchor="middle"
          dominantBaseline="middle"
          fill={color}
          fontFamily={FONT_MONO}
          fontSize={7}
          fontWeight={600}
          letterSpacing="0.05em"
        >
          {outcomeBand(pct).toUpperCase()}
        </text>
      </svg>
    </div>
  );
}
