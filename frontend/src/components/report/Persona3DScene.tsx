import { useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import { useSessionStore } from "@/store/sessionStore";
import { COLORS } from "@/lib/theme";
import type { PersonaPoint } from "@/types/events";

const CLUSTER_HEX = [COLORS.alert, COLORS.orange, COLORS.positive, COLORS.teal];
const CLUSTER_LABELS = ["Skeptics", "Pragmatists", "Enthusiasts"];

/**
 * Interactive three.js point-cloud of the 1,500-persona PCA projection, colored
 * by cluster. Richer and more tactile than the Plotly scatter3d - orbit, zoom,
 * pan. `preserveDrawingBuffer` lets the report exporter snapshot the canvas.
 */
function PointCloud({ points }: { points: PersonaPoint[] }) {
  const geometry = useMemo(() => {
    const geom = new THREE.BufferGeometry();
    const n = points.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);

    let maxAbs = 1e-6;
    for (const p of points) {
      for (let k = 0; k < 3; k++) maxAbs = Math.max(maxAbs, Math.abs(p.pca[k]));
    }
    const scale = 4 / maxAbs;
    const color = new THREE.Color();
    points.forEach((p, i) => {
      positions[i * 3] = p.pca[0] * scale;
      positions[i * 3 + 1] = p.pca[1] * scale;
      positions[i * 3 + 2] = p.pca[2] * scale;
      color.set(CLUSTER_HEX[p.cluster] ?? COLORS.teal);
      colors[i * 3] = color.r;
      colors[i * 3 + 1] = color.g;
      colors[i * 3 + 2] = color.b;
    });
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    return geom;
  }, [points]);

  return (
    <points geometry={geometry}>
      <pointsMaterial size={0.12} vertexColors sizeAttenuation transparent opacity={0.85} />
    </points>
  );
}

export function Persona3DScene({ points: override }: { points?: PersonaPoint[] }) {
  const storePoints = useSessionStore((s) => s.personaPoints);
  const points = override ?? storePoints;

  if (!points.length) {
    return (
      <div className="flex h-full items-center justify-center font-mono text-xs text-muted">
        awaiting persona vectors
      </div>
    );
  }

  return (
    <div className="relative h-full w-full">
      <Canvas
        camera={{ position: [6, 5, 6], fov: 50 }}
        gl={{ preserveDrawingBuffer: true, antialias: true }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={0.8} />
        <PointCloud points={points} />
        <OrbitControls enablePan enableZoom enableRotate autoRotate autoRotateSpeed={0.6} />
      </Canvas>
      <div className="pointer-events-none absolute left-2 top-1 flex gap-3 font-mono text-2xs">
        {CLUSTER_LABELS.map((l, i) => (
          <span key={l} style={{ color: CLUSTER_HEX[i] }}>
            {l}
          </span>
        ))}
      </div>
    </div>
  );
}
