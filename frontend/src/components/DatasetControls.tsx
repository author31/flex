import { EditParams, ModelInfo } from "../api/client";

// Model + params selectors that apply to the whole dataset, plus the action buttons.
export function DatasetControls({
  models,
  modelKey,
  onModelKey,
  params,
  onParams,
  onSave,
  onRun,
  canSave,
  canRun,
  saving,
  running,
  saveLabel = "Save dataset",
}: {
  models: ModelInfo[];
  modelKey: string;
  onModelKey: (k: string) => void;
  params: EditParams;
  onParams: (p: EditParams) => void;
  onSave: () => void;
  onRun: () => void;
  canSave: boolean;
  canRun: boolean;
  saving: boolean;
  running: boolean;
  saveLabel?: string;
}) {
  return (
    <section className="card toolbar" aria-label="Dataset controls">
      <div className="field">
        <label htmlFor="ctl-model">Model</label>
        <select id="ctl-model" value={modelKey} onChange={(e) => onModelKey(e.target.value)} aria-label="Model">
          {models.map((m) => (
            <option key={m.key} value={m.key}>
              {m.key}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor="ctl-steps">Steps</label>
        <input
          id="ctl-steps"
          type="number"
          min={10}
          max={50}
          value={params.steps ?? 25}
          onChange={(e) => onParams({ ...params, steps: Number(e.target.value) })}
          aria-label="Steps"
          style={{ width: 84 }}
        />
      </div>
      <div className="field">
        <label htmlFor="ctl-guidance">Guidance</label>
        <input
          id="ctl-guidance"
          type="number"
          step={0.5}
          min={1}
          max={20}
          value={params.guidance ?? 8}
          onChange={(e) => onParams({ ...params, guidance: Number(e.target.value) })}
          aria-label="Guidance"
          style={{ width: 84 }}
        />
      </div>
      <span className="spacer" />
      <button type="button" onClick={onSave} disabled={!canSave || saving}>
        {saving ? "Saving…" : saveLabel}
      </button>
      <button type="button" className="btn-primary" onClick={onRun} disabled={!canRun || running}>
        {running ? "Running…" : "Run batch edit"}
      </button>
    </section>
  );
}
