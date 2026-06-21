import { createElement, useEffect } from "react";

// 3D viewer for a GLB mesh. Uses the <model-viewer> web component, loaded lazily
// in the browser (skipped/harmless in jsdom). Rendered via createElement to avoid
// global JSX intrinsic-element augmentation.
export function MeshViewer({ url }: { url: string }) {
  useEffect(() => {
    import("@google/model-viewer").catch(() => undefined);
  }, []);

  return (
    <section className="card" aria-label="3D mesh">
      {createElement("model-viewer", {
        src: url,
        "camera-controls": true,
        "auto-rotate": true,
        "shadow-intensity": "1",
        "data-testid": "mesh-viewer",
        style: { width: "100%", height: 420, background: "#111111", borderRadius: 8 },
      })}
      <p className="secondary" style={{ marginTop: 8 }}>
        Drag to orbit · scroll to zoom ·{" "}
        <a href={url} download>
          download .glb
        </a>
      </p>
    </section>
  );
}
