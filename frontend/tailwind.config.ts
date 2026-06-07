import type { Config } from "tailwindcss";

// Bloomberg-dark palette is the single source of truth for the design system.
// Values mirror the spec (section 10.2) and are also exposed as CSS variables
// in src/styles/tokens.css for non-Tailwind consumers (D3 / Plotly).
const config: Config = {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0A0E1A",
        panel: "#0F1923",
        "panel-raised": "#13212E",
        teal: "#00B4D8",
        orange: "#F5A623",
        positive: "#00E676",
        alert: "#FF4C4C",
        data: "#E8ECF0",
        muted: "#546E7A",
        grid: "#1B2A38",
      },
      fontFamily: {
        mono: ["'IBM Plex Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
        display: ["'Archivo'", "system-ui", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.625rem", { lineHeight: "0.875rem" }],
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(255,255,255,0.03) inset, 0 8px 24px -12px rgba(0,0,0,0.8)",
        glow: "0 0 0 1px rgba(0,180,216,0.35), 0 0 18px -2px rgba(0,180,216,0.4)",
      },
      keyframes: {
        "panel-in": {
          "0%": { opacity: "0", transform: "translateY(8px) scale(0.995)" },
          "100%": { opacity: "1", transform: "translateY(0) scale(1)" },
        },
        "pulse-stream": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.35" },
        },
        "ticker": {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
        "scan": {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
      },
      animation: {
        "panel-in": "panel-in 0.5s cubic-bezier(0.16, 1, 0.3, 1) both",
        "pulse-stream": "pulse-stream 1.4s ease-in-out infinite",
        ticker: "ticker 40s linear infinite",
        scan: "scan 7s linear infinite",
      },
    },
  },
  plugins: [],
};

export default config;
