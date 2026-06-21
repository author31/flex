import { useEffect, useMemo, useState } from "react";
import { Comparison, CreateEditRequest, ImageOut, api } from "../api/client";
import { ImageUploader } from "../components/ImageUploader";
import { RegionSelector, Selection } from "../components/RegionSelector";
import { RegionPreview } from "../components/RegionPreview";
import { PromptPanel, PromptState } from "../components/PromptPanel";
import { SubmitBar } from "../components/SubmitBar";
import { ComparisonView } from "../components/ComparisonView";
import { MeshExport } from "../components/MeshExport";
import { useCreateEdit } from "../hooks/useCreateEdit";
import { useEditPolling } from "../hooks/useEditPolling";
import { useModels } from "../hooks/useModels";

export function EditorPage() {
  const [image, setImage] = useState<ImageOut | null>(null);
  const [selection, setSelection] = useState<Selection>(null);
  const [promptState, setPromptState] = useState<PromptState | null>(null);
  const [editId, setEditId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);
  const [expert, setExpert] = useState(false);
  const [modelKey, setModelKey] = useState<string>("");

  const createEdit = useCreateEdit();
  const poll = useEditPolling(editId);
  const models = useModels();

  // Default the checkpoint selection once models load.
  useEffect(() => {
    if (models.data && !modelKey) setModelKey(models.data.default);
  }, [models.data, modelKey]);

  const hasPrompt = !!(promptState?.prompt?.trim() || promptState?.preset);
  const canSubmit = !!image && !!selection && hasPrompt;
  const running = createEdit.isPending || poll.data?.status === "processing";

  useEffect(() => {
    if (poll.data?.status === "completed" && editId) {
      api.getComparison(editId).then(setComparison).catch(() => undefined);
    }
  }, [poll.data?.status, editId]);

  const jobError =
    poll.data?.status === "failed"
      ? poll.data.error ?? "edit failed"
      : (createEdit.error as Error | undefined)?.message ?? null;

  function reset() {
    setSelection(null);
    setPromptState(null);
    setEditId(null);
    setComparison(null);
  }

  function submit() {
    if (!image || !selection || !promptState) return;
    setComparison(null);
    const body: CreateEditRequest = {
      image_id: image.image_id,
      box: selection.box,
      params: promptState.params,
      ...(modelKey ? { model: modelKey } : {}),
      ...(promptState.preset ? { preset: promptState.preset } : { prompt: promptState.prompt }),
    };
    createEdit.mutate(body, { onSuccess: (r) => setEditId(r.edit_id) });
  }

  const header = useMemo(
    () => (
      <header style={{ padding: "24px 0" }}>
        <h1 style={{ margin: 0 }}>Facial Expression Editor</h1>
        <p className="muted" style={{ marginTop: 4 }}>
          Upload → select region → prompt → compare
        </p>
      </header>
    ),
    [],
  );

  return (
    <main style={{ maxWidth: 920, margin: "0 auto", padding: "0 16px 64px" }}>
      {header}

      {!image ? (
        <ImageUploader
          onUploaded={(img) => {
            setImage(img);
            reset();
          }}
        />
      ) : (
        <div style={{ display: "grid", gap: 16 }}>
          <RegionSelector image={image} onSelection={setSelection} />
          {selection && <RegionPreview image={image} box={selection.box} />}
          <PromptPanel onChange={setPromptState} />
          {(models.data?.models.length ?? 0) > 1 && (
            <label style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <span className="muted">Checkpoint</span>
              <select value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
                {models.data!.models.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.key}
                    {m.lora ? " (LoRA)" : ""}
                    {m.key === models.data!.default ? " — default" : ""}
                  </option>
                ))}
              </select>
            </label>
          )}
          <SubmitBar canSubmit={canSubmit} pending={running} onSubmit={submit} error={jobError} />

          {running && <p className="muted">Working… this can take a moment on first run.</p>}

          {comparison && (
            <>
              <ComparisonView comparison={comparison} />

              {/* Expert: 3D export, offered after a result exists */}
              <label
                className="muted"
                style={{ display: "flex", gap: 8, alignItems: "center" }}
              >
                <input
                  type="checkbox"
                  checked={expert}
                  onChange={(e) => setExpert(e.target.checked)}
                />
                Expert options — export the result to a 3D mesh
              </label>
              {expert && editId && <MeshExport editId={editId} />}
            </>
          )}

          <button
            type="button"
            onClick={() => {
              setImage(null);
              reset();
            }}
          >
            Start over with a new image
          </button>
        </div>
      )}
    </main>
  );
}
