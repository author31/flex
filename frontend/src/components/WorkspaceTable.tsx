import { ImageOut, MetricRecordOut } from "../api/client";
import { RegionSelector, Selection } from "./RegionSelector";
import { MetricsCell } from "./MetricsCell";

export interface WorkspaceRow {
  image: ImageOut;
  box: [number, number, number, number] | null;
  prompt: string;
}

// CSS-cropped thumbnail of the selected region (no extra request).
function RegionThumb({ image, box }: { image: ImageOut; box: [number, number, number, number] }) {
  const [x, y, w, h] = box;
  const maxW = 132;
  const scale = w > 0 ? Math.min(1, maxW / w) : 1;
  return (
    <div
      data-testid="region-thumb"
      aria-label="Selected region preview"
      style={{
        width: Math.max(1, Math.round(w * scale)),
        height: Math.max(1, Math.round(h * scale)),
        backgroundImage: `url(${image.url})`,
        backgroundRepeat: "no-repeat",
        backgroundSize: `${Math.round(image.width * scale)}px ${Math.round(image.height * scale)}px`,
        backgroundPosition: `-${Math.round(x * scale)}px -${Math.round(y * scale)}px`,
        border: "1px solid var(--border-light)",
        borderRadius: 8,
        flex: "0 0 auto",
      }}
    />
  );
}

// One row per uploaded image: image (drag to pick region) on the left; the selected
// region preview sitting next to the prompt input on the right; metrics after a run.
export function WorkspaceTable({
  rows,
  onBox,
  onPrompt,
  recordForRow,
  hasRun,
}: {
  rows: WorkspaceRow[];
  onBox: (index: number, box: Selection) => void;
  onPrompt: (index: number, prompt: string) => void;
  recordForRow: (index: number) => MetricRecordOut | null;
  hasRun: boolean;
}) {
  return (
    <div role="table" aria-label="Study dataset" style={{ display: "grid", gap: 14 }}>
      {rows.map((row, i) => (
        <div key={row.image.image_id} role="row" data-testid="workspace-row" className="row-card">
          <div style={{ position: "relative" }}>
            <span className="row-index">#{i + 1}</span>
            <RegionSelector image={row.image} onSelection={(s) => onBox(i, s)} />
          </div>

          <div className="row-side">
            <div style={{ display: "flex", gap: 14, alignItems: "flex-start", flexWrap: "wrap" }}>
              <div className="field">
                <label>Selected region</label>
                {row.box ? (
                  <RegionThumb image={row.image} box={row.box} />
                ) : (
                  <div
                    className="empty"
                    style={{ padding: 16, minWidth: 132, fontSize: 13 }}
                    data-testid="region-empty"
                  >
                    Drag on the image →
                  </div>
                )}
                {row.box && (
                  <span className="muted" style={{ fontSize: 12 }}>
                    {row.box[2]} × {row.box[3]} px
                  </span>
                )}
              </div>

              <div className="field" style={{ flex: 1, minWidth: 220 }}>
                <label htmlFor={`prompt-${i}`}>Prompt</label>
                <textarea
                  id={`prompt-${i}`}
                  aria-label={`Prompt for image ${i + 1}`}
                  placeholder="describe the expression edit, e.g. a wide happy smile"
                  value={row.prompt}
                  onChange={(e) => onPrompt(i, e.target.value)}
                  rows={3}
                  style={{ width: "100%" }}
                />
              </div>
            </div>

            {hasRun &&
              (() => {
                const rec = recordForRow(i);
                return (
                  <div style={{ display: "flex", gap: 14, alignItems: "flex-start", flexWrap: "wrap" }}>
                    <div className="field">
                      <label>Edited result</label>
                      {rec?.result_url ? (
                        <img
                          data-testid="result-image"
                          src={rec.result_url}
                          alt={`edited result ${i + 1}`}
                          style={{
                            maxWidth: 160,
                            borderRadius: 8,
                            border: "1px solid var(--border-light)",
                          }}
                        />
                      ) : (
                        <div className="empty" style={{ padding: 16, minWidth: 132, fontSize: 13 }}>
                          {rec?.status === "failed" ? "no result" : "rendering…"}
                        </div>
                      )}
                    </div>
                    <div className="field">
                      <label>Metrics</label>
                      <MetricsCell record={rec} />
                    </div>
                  </div>
                );
              })()}
          </div>
        </div>
      ))}
    </div>
  );
}
