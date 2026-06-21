import { useMemo } from "react";
import { DatasetOut, ModelInfo, RevisionOut } from "../api/client";
import { MetricsCell } from "./MetricsCell";

// One side of the two-model comparison: pick a model, run the dataset, show per-row
// result images + metrics. Rows are aligned by dataset item across both columns.
export function CompareColumn({
  title,
  models,
  modelKey,
  onModelKey,
  dataset,
  revision,
  onRun,
  running,
}: {
  title: string;
  models: ModelInfo[];
  modelKey: string;
  onModelKey: (k: string) => void;
  dataset: DatasetOut | null;
  revision?: RevisionOut | null;
  onRun: () => void;
  running: boolean;
}) {
  const byItem = useMemo(() => {
    const map = new Map(revision?.records.map((r) => [r.dataset_item_id, r]) ?? []);
    return map;
  }, [revision]);

  return (
    <section className="card" aria-label={title} style={{ flex: 1, display: "grid", gap: 12 }}>
      <header style={{ display: "flex", gap: 8, alignItems: "center", justifyContent: "space-between" }}>
        <strong>{title}</strong>
        <span className="muted">{revision ? `revision ${revision.number} (${revision.status})` : ""}</span>
      </header>
      <div className="toolbar">
        <div className="field">
          <label>Model</label>
          <select value={modelKey} onChange={(e) => onModelKey(e.target.value)} aria-label={`${title} model`}>
            {models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.key}
              </option>
            ))}
          </select>
        </div>
        <span className="spacer" />
        <button type="button" className="btn-primary" onClick={onRun} disabled={!dataset || running}>
          {running ? "Running…" : "Run"}
        </button>
      </div>

      {dataset?.items.map((item, i) => {
        const rec = byItem.get(item.id);
        return (
          <div key={item.id} data-testid="compare-row" style={{ display: "grid", gap: 4, borderTop: "1px solid var(--border)", paddingTop: 8 }}>
            <span className="muted">#{i + 1} · {item.prompt}</span>
            {rec?.result_url ? (
              <img src={rec.result_url} alt={`result ${i + 1} (${title})`} style={{ maxWidth: 160 }} />
            ) : (
              <img src={item.image_url} alt={`source ${i + 1}`} style={{ maxWidth: 160, opacity: 0.6 }} />
            )}
            <MetricsCell record={rec} />
          </div>
        );
      })}
    </section>
  );
}
