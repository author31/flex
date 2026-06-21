import { MetricRecordOut } from "../api/client";

// Renders the five evaluator metrics for one row, or a pending / failed state.
export function MetricsCell({ record }: { record?: MetricRecordOut | null }) {
  if (!record) return <span className="muted">—</span>;
  if (record.status === "failed") {
    return (
      <span role="alert" style={{ color: "var(--danger)" }}>
        ✗ failed: {record.error ?? "unknown error"}
      </span>
    );
  }
  if (record.status === "processing" || !record.metrics) {
    return <span className="muted">working…</span>;
  }
  const m = record.metrics;
  const ok = m.edit_success;
  return (
    <dl data-testid="metrics" className="metric-grid">
      <span>CLIP (in-mask)</span>
      <b>{m.clip_similarity_in_mask.toFixed(3)}</b>
      <span>edit success</span>
      <b style={{ color: ok ? "var(--ok)" : "var(--danger)" }}>
        {ok ? "✓" : "✗"} ({m.edit_success_score.toFixed(3)})
      </b>
      <span>identity</span>
      <b style={{ color: m.identity_preserved ? "var(--ok)" : "var(--danger)" }}>
        {m.identity_preserved ? "✓ preserved" : "✗"}
      </b>
      <span>latency</span>
      <b>{m.latency_ms} ms</b>
    </dl>
  );
}
