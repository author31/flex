import { useState } from "react";
import { Comparison } from "../api/client";
import { MetricsPanel } from "./MetricsPanel";

// Before/after drag slider + metrics.
export function ComparisonView({ comparison }: { comparison: Comparison }) {
  const [pct, setPct] = useState(50);
  return (
    <section className="card" aria-label="Comparison">
      <div style={{ position: "relative", userSelect: "none" }}>
        <img src={comparison.original.url} alt="original" style={{ width: "100%", display: "block" }} />
        <div
          style={{
            position: "absolute",
            inset: 0,
            width: `${pct}%`,
            overflow: "hidden",
            borderRight: "2px solid var(--color-fg)",
          }}
        >
          <img
            src={comparison.edited.url}
            alt="edited"
            style={{ width: `${(100 / pct) * 100}%`, maxWidth: "none", display: "block" }}
          />
        </div>
      </div>
      <input
        aria-label="Compare slider"
        type="range"
        min={0}
        max={100}
        value={pct}
        onChange={(e) => setPct(Number(e.target.value))}
        style={{ width: "100%", marginTop: 8 }}
      />
      <p className="secondary">
        Prompt: <em>{comparison.prompt}</em> · region: {comparison.region}
      </p>
      <MetricsPanel metrics={comparison.metrics} />
    </section>
  );
}
