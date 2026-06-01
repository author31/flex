import { useRef, useState } from "react";

type Box = [number, number, number, number]; // [x, y, w, h] in image-pixel space
interface LocalRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

// Drag a rectangle over the image to pick the edit region. Emits the box in the
// image's natural pixel space; the backend turns it into the mask.
export function DragSelect({
  imageUrl,
  width,
  height,
  onChange,
}: {
  imageUrl: string;
  width: number;
  height: number;
  onChange: (box: Box | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const start = useRef<{ x: number; y: number } | null>(null);
  const [rect, setRect] = useState<LocalRect | null>(null);

  function local(clientX: number, clientY: number) {
    const r = ref.current!.getBoundingClientRect();
    return { x: clientX - r.left, y: clientY - r.top };
  }

  function normalize(a: { x: number; y: number }, b: { x: number; y: number }): LocalRect {
    return {
      x: Math.min(a.x, b.x),
      y: Math.min(a.y, b.y),
      w: Math.abs(a.x - b.x),
      h: Math.abs(a.y - b.y),
    };
  }

  function toBox(r: LocalRect): Box {
    const box = ref.current!.getBoundingClientRect();
    const sx = box.width ? width / box.width : 1;
    const sy = box.height ? height / box.height : 1;
    const clamp = (v: number, max: number) => Math.max(0, Math.min(max, Math.round(v)));
    const x = clamp(r.x * sx, width);
    const y = clamp(r.y * sy, height);
    const w = clamp(r.w * sx, width - x);
    const h = clamp(r.h * sy, height - y);
    return [x, y, w, h];
  }

  function onMouseDown(e: React.MouseEvent) {
    const p = local(e.clientX, e.clientY);
    start.current = p;
    setRect({ x: p.x, y: p.y, w: 0, h: 0 });
    onChange(null);
  }

  function onMouseMove(e: React.MouseEvent) {
    if (!start.current) return;
    setRect(normalize(start.current, local(e.clientX, e.clientY)));
  }

  function finish(e: React.MouseEvent) {
    if (!start.current) return;
    const r = normalize(start.current, local(e.clientX, e.clientY));
    start.current = null;
    onChange(r.w > 2 && r.h > 2 ? toBox(r) : null);
    setRect(null); // no persistent highlight once the region is selected
  }

  return (
    <div>
      <div
        ref={ref}
        data-testid="drag-surface"
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={finish}
        onMouseLeave={(e) => start.current && finish(e)}
        style={{ position: "relative", maxWidth: "100%", cursor: "crosshair", userSelect: "none" }}
      >
        <img src={imageUrl} alt="to edit" draggable={false} style={{ width: "100%", display: "block" }} />
        {rect && rect.w > 0 && rect.h > 0 && (
          <div
            style={{
              position: "absolute",
              left: rect.x,
              top: rect.y,
              width: rect.w,
              height: rect.h,
              border: "2px solid var(--color-fg)",
              boxShadow: "0 0 0 9999px rgba(17,17,17,0.45)",
              pointerEvents: "none",
            }}
          />
        )}
      </div>
      <p className="muted" style={{ marginTop: 8 }}>
        Drag a rectangle over the area you want to edit.
      </p>
    </div>
  );
}
