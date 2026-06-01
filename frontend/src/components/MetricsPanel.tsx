import { Metrics } from "../api/client";

export function MetricsPanel({ metrics }: { metrics: Metrics }) {
  const rows: [string, string][] = [
    ["CLIP similarity (in mask)", metrics.clip_similarity_in_mask.toFixed(3)],
    ["Edit success", metrics.edit_success ? "yes" : "no"],
    ["Edit success score", metrics.edit_success_score.toFixed(3)],
    ["Identity preserved", metrics.identity_preserved ? "yes" : "no"],
    ["Latency", `${metrics.latency_ms} ms`],
  ];
  return (
    <table aria-label="Metrics" style={{ width: "100%", borderCollapse: "collapse" }}>
      <tbody>
        {rows.map(([k, v]) => (
          <tr key={k}>
            <td className="secondary" style={{ padding: "4px 8px" }}>{k}</td>
            <td style={{ padding: "4px 8px", textAlign: "right", fontWeight: 600 }}>{v}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
