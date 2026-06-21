import { useEffect, useMemo, useState } from "react";
import { api, DatasetOut, EditParams, ImageOut, MetricRecordOut } from "../api/client";
import { ImageUploader } from "../components/ImageUploader";
import { WorkspaceRow, WorkspaceTable } from "../components/WorkspaceTable";
import { DatasetControls } from "../components/DatasetControls";
import { useModels } from "../hooks/useModels";
import { useDatasets } from "../hooks/useDatasets";
import { useCreateDataset } from "../hooks/useCreateDataset";
import { useUpdateDataset } from "../hooks/useUpdateDataset";
import { useRunBatch } from "../hooks/useRunBatch";
import { useRevision } from "../hooks/useRevision";

export function WorkspacePage() {
  const [rows, setRows] = useState<WorkspaceRow[]>([]);
  const [dataset, setDataset] = useState<DatasetOut | null>(null);
  const [revNumber, setRevNumber] = useState<number | null>(null);
  const [modelKey, setModelKey] = useState("");
  const [params, setParams] = useState<EditParams>({ steps: 25, guidance: 8.0 });
  const [loading, setLoading] = useState(false);

  const models = useModels();
  const datasets = useDatasets();
  const create = useCreateDataset();
  const update = useUpdateDataset();
  const run = useRunBatch();
  const revision = useRevision(dataset?.dataset_id ?? null, revNumber);

  useEffect(() => {
    if (models.data && !modelKey) setModelKey(models.data.default);
  }, [models.data, modelKey]);

  const canSave = rows.length > 0 && rows.every((r) => r.box && r.prompt.trim());
  const canRun = !!dataset;

  function addImage(img: ImageOut) {
    setRows((rs) => [...rs, { image: img, box: null, prompt: "" }]);
    setRevNumber(null); // editing the set invalidates the shown revision metrics
  }
  const setBox = (i: number, s: { box: [number, number, number, number] } | null) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, box: s ? s.box : null } : r)));
  const setPrompt = (i: number, prompt: string) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, prompt } : r)));

  function newDataset() {
    setRows([]);
    setDataset(null);
    setRevNumber(null);
    setParams({ steps: 25, guidance: 8.0 });
    if (models.data) setModelKey(models.data.default);
  }

  async function loadDataset(id: string) {
    if (!id) return;
    setLoading(true);
    try {
      const ds = await api.getDataset(id);
      setRows(
        ds.items.map((it) => ({
          image: {
            image_id: it.image_id,
            url: it.image_url,
            width: it.image_width ?? 0,
            height: it.image_height ?? 0,
          },
          box: it.box ?? null,
          prompt: it.prompt,
        })),
      );
      setModelKey(ds.model_key);
      setParams(ds.params);
      setDataset(ds);
      setRevNumber(ds.revisions.length ? Math.max(...ds.revisions.map((r) => r.number)) : null);
    } finally {
      setLoading(false);
    }
  }

  function save() {
    const body = {
      name: dataset?.name,
      model_key: modelKey,
      params,
      items: rows.map((r) => ({ image_id: r.image.image_id, box: r.box!, prompt: r.prompt.trim() })),
    };
    if (dataset) {
      update.mutate({ datasetId: dataset.dataset_id, body }, { onSuccess: setDataset });
    } else {
      create.mutate(body, { onSuccess: setDataset });
    }
  }

  function runBatch() {
    if (!dataset) return;
    run.mutate(
      { datasetId: dataset.dataset_id, modelKey },
      { onSuccess: (r) => setRevNumber(r.number) },
    );
  }

  const recordByItem = useMemo(() => {
    const map = new Map<string, MetricRecordOut>();
    revision.data?.records.forEach((rec) => map.set(rec.dataset_item_id, rec));
    return map;
  }, [revision.data]);

  const recordForRow = (i: number): MetricRecordOut | null => {
    const item = dataset?.items[i];
    return item ? recordByItem.get(item.id) ?? null : null;
  };

  const hasRun = revNumber != null;
  const saving = create.isPending || update.isPending;
  const error =
    (create.error as Error | undefined)?.message ??
    (update.error as Error | undefined)?.message ??
    (run.error as Error | undefined)?.message;

  return (
    <main className="page">
      <header className="page-header">
        <h1 style={{ margin: 0 }}>Study Workspace</h1>
        <p>Build a new study or load a saved one → edit regions + prompts → save/update → run batch edit.</p>
      </header>

      <div style={{ display: "grid", gap: 16 }}>
        <section className="card toolbar" aria-label="Dataset library">
          <div className="field" style={{ minWidth: 260 }}>
            <label htmlFor="load-dataset">Load saved dataset</label>
            <select
              id="load-dataset"
              aria-label="Load saved dataset"
              value={dataset?.dataset_id ?? ""}
              onChange={(e) => loadDataset(e.target.value)}
            >
              <option value="">— new dataset —</option>
              {datasets.data?.datasets.map((d) => (
                <option key={d.dataset_id} value={d.dataset_id}>
                  {d.name} ({d.item_count} img{d.latest_revision ? `, rev ${d.latest_revision}` : ""})
                </option>
              ))}
            </select>
          </div>
          <span className="spacer" />
          <button type="button" onClick={newDataset} disabled={loading}>
            + New dataset
          </button>
        </section>

        <ImageUploader onUploaded={addImage} />

        {rows.length === 0 ? (
          <div className="empty">
            {loading ? "Loading dataset…" : "No images yet — upload a few, or load a saved dataset above."}
          </div>
        ) : (
          <>
            <DatasetControls
              models={models.data?.models ?? []}
              modelKey={modelKey}
              onModelKey={setModelKey}
              params={params}
              onParams={setParams}
              onSave={save}
              onRun={runBatch}
              canSave={canSave}
              canRun={canRun}
              saving={saving}
              running={run.isPending || revision.data?.status === "processing"}
              saveLabel={dataset ? "Update dataset" : "Save dataset"}
            />

            <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
              <span className="pill">{rows.length} image{rows.length === 1 ? "" : "s"}</span>
              {dataset && (
                <span className="pill" data-testid="dataset-saved">
                  {dataset.name} · {dataset.dataset_id}
                </span>
              )}
              {hasRun && revision.data && (
                <span className="pill">
                  revision {revNumber} · {revision.data.status}
                </span>
              )}
              {!canSave && (
                <span className="muted" style={{ fontSize: 13 }}>
                  every row needs a region + prompt
                </span>
              )}
            </div>

            {error && <p role="alert" style={{ color: "var(--danger)" }}>{error}</p>}

            <WorkspaceTable
              rows={rows}
              onBox={setBox}
              onPrompt={setPrompt}
              recordForRow={recordForRow}
              hasRun={hasRun}
            />
          </>
        )}
      </div>
    </main>
  );
}
