import { useEffect, useState } from "react";
import { CompareColumn } from "../components/CompareColumn";
import { useModels } from "../hooks/useModels";
import { useDatasets, useDataset } from "../hooks/useDatasets";
import { useRunBatch } from "../hooks/useRunBatch";
import { useRevision } from "../hooks/useRevision";

// Two-model comparison over a previously saved dataset (base vs finetuned).
export function ComparePage() {
  const [datasetId, setDatasetId] = useState("");
  const [leftKey, setLeftKey] = useState("");
  const [rightKey, setRightKey] = useState("");
  const [leftRev, setLeftRev] = useState<number | null>(null);
  const [rightRev, setRightRev] = useState<number | null>(null);

  const models = useModels();
  const datasets = useDatasets();
  const dataset = useDataset(datasetId || null);
  const runLeft = useRunBatch();
  const runRight = useRunBatch();
  const leftRevision = useRevision(datasetId || null, leftRev);
  const rightRevision = useRevision(datasetId || null, rightRev);

  useEffect(() => {
    if (!models.data) return;
    const keys = models.data.models.map((m) => m.key);
    if (!leftKey) setLeftKey(models.data.default);
    if (!rightKey) setRightKey(keys.find((k) => k !== models.data!.default) ?? keys[0]);
  }, [models.data, leftKey, rightKey]);

  function pickDataset(id: string) {
    setDatasetId(id);
    setLeftRev(null);
    setRightRev(null);
  }

  const ds = dataset.data ?? null;
  const modelList = models.data?.models ?? [];

  return (
    <main className="page">
      <header className="page-header">
        <h1 style={{ margin: 0 }}>Compare Models</h1>
        <p>Pick a saved dataset, choose a model per side, run both — base vs finetuned, side by side.</p>
      </header>

      <section className="card" style={{ maxWidth: 460 }}>
        <div className="field">
          <label htmlFor="cmp-dataset">Dataset</label>
          <select id="cmp-dataset" value={datasetId} onChange={(e) => pickDataset(e.target.value)} aria-label="Dataset">
            <option value="">— select a saved dataset —</option>
            {datasets.data?.datasets.map((d) => (
              <option key={d.dataset_id} value={d.dataset_id}>
                {d.name} ({d.item_count} images)
              </option>
            ))}
          </select>
        </div>
      </section>

      {ds && (
        <div style={{ display: "flex", gap: 16, marginTop: 16, alignItems: "start" }}>
          <CompareColumn
            title="Model A"
            models={modelList}
            modelKey={leftKey}
            onModelKey={setLeftKey}
            dataset={ds}
            revision={leftRevision.data}
            running={runLeft.isPending || leftRevision.data?.status === "processing"}
            onRun={() =>
              runLeft.mutate(
                { datasetId, modelKey: leftKey },
                { onSuccess: (r) => setLeftRev(r.number) },
              )
            }
          />
          <CompareColumn
            title="Model B"
            models={modelList}
            modelKey={rightKey}
            onModelKey={setRightKey}
            dataset={ds}
            revision={rightRevision.data}
            running={runRight.isPending || rightRevision.data?.status === "processing"}
            onRun={() =>
              runRight.mutate(
                { datasetId, modelKey: rightKey },
                { onSuccess: (r) => setRightRev(r.number) },
              )
            }
          />
        </div>
      )}
    </main>
  );
}
