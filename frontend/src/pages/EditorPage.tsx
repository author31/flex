import { useEffect, useMemo, useState } from "react";
import { Comparison, CreateEditRequest, ImageOut, api } from "../api/client";
import { ImageUploader } from "../components/ImageUploader";
import { RegionSelector, Selection } from "../components/RegionSelector";
import { PromptPanel, PromptState } from "../components/PromptPanel";
import { SubmitBar } from "../components/SubmitBar";
import { ComparisonView } from "../components/ComparisonView";
import { useCreateEdit } from "../hooks/useCreateEdit";
import { useEditPolling } from "../hooks/useEditPolling";

export function EditorPage() {
  const [image, setImage] = useState<ImageOut | null>(null);
  const [selection, setSelection] = useState<Selection>(null);
  const [promptState, setPromptState] = useState<PromptState | null>(null);
  const [editId, setEditId] = useState<string | null>(null);
  const [comparison, setComparison] = useState<Comparison | null>(null);

  const createEdit = useCreateEdit();
  const poll = useEditPolling(editId);

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
          <PromptPanel onChange={setPromptState} />
          <SubmitBar canSubmit={canSubmit} pending={running} onSubmit={submit} error={jobError} />

          {running && <p className="muted">Working… this can take a moment on first run.</p>}
          {comparison && <ComparisonView comparison={comparison} />}

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
