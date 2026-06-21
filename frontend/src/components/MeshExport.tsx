import { useState } from "react";
import { useCreateMesh, useMeshPolling } from "../hooks/useMesh";
import { MeshViewer } from "./MeshViewer";

// Expert feature: turn the edited result into a 3D mesh and view it.
export function MeshExport({ editId }: { editId: string }) {
  const create = useCreateMesh();
  const [meshId, setMeshId] = useState<string | null>(null);
  const poll = useMeshPolling(meshId);

  const running = create.isPending || poll.data?.status === "processing";
  const url = poll.data?.status === "completed" ? (poll.data.url ?? null) : null;
  const error =
    poll.data?.status === "failed"
      ? (poll.data.error ?? "mesh generation failed")
      : ((create.error as Error | undefined)?.message ?? null);

  return (
    <section className="card" aria-label="Expert 3D export">
      <button
        type="button"
        disabled={running}
        onClick={() => create.mutate(editId, { onSuccess: (r) => setMeshId(r.mesh_id) })}
      >
        {running ? "Generating mesh…" : "Export to 3D mesh"}
      </button>
      {running && <p className="muted" style={{ marginTop: 8 }}>This can take a while on first run.</p>}
      {error && (
        <p role="alert" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
      {url && (
        <div style={{ marginTop: 12 }}>
          <MeshViewer url={url} />
        </div>
      )}
    </section>
  );
}
