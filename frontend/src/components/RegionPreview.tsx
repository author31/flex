import { ImageOut } from "../api/client";

// Shows the currently selected region as a cropped thumbnail (CSS-cropped from the
// source image) next to the prompt input — feedback without highlighting the image.
export function RegionPreview({
  image,
  box,
}: {
  image: ImageOut;
  box: [number, number, number, number];
}) {
  const [x, y, w, h] = box;
  const maxW = 200;
  const scale = w > 0 ? Math.min(1, maxW / w) : 1;

  return (
    <section className="card" aria-label="Selected region" style={{ display: "flex", gap: 12 }}>
      <div
        data-testid="region-preview"
        style={{
          width: Math.max(1, Math.round(w * scale)),
          height: Math.max(1, Math.round(h * scale)),
          backgroundImage: `url(${image.url})`,
          backgroundRepeat: "no-repeat",
          backgroundSize: `${Math.round(image.width * scale)}px ${Math.round(image.height * scale)}px`,
          backgroundPosition: `-${Math.round(x * scale)}px -${Math.round(y * scale)}px`,
          border: "1px solid var(--border)",
          borderRadius: 6,
          flex: "0 0 auto",
        }}
      />
      <div className="secondary">
        <strong>Selected region</strong>
        <div className="muted">
          x {x}, y {y} · {w} × {h} px
        </div>
      </div>
    </section>
  );
}
