import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

// The FastAPI backend is expected on :8000.
// We proxy /api so the browser EventSource can hit a same-origin URL and
// avoid CORS during development.
const BACKEND_URL = process.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "src"),
    },
  },
  server: {
    port: 3000,
    strictPort: true,
    proxy: {
      "/api": {
        target: BACKEND_URL,
        changeOrigin: true,
        // SSE requires the connection to stay open; disable buffering.
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyReq) => {
            proxyReq.setHeader("Cache-Control", "no-cache");
          });
        },
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
    chunkSizeWarningLimit: 900,
    rollupOptions: {
      output: {
        // Split the heavy visualization vendors so the app shell and panels
        // can be cached independently of Plotly/D3.
        manualChunks: {
          plotly: ["plotly.js-dist-min", "react-plotly.js"],
          d3: ["d3"],
          react: ["react", "react-dom"],
        },
      },
    },
  },
});
