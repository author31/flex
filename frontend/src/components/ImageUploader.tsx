import { useCallback, useRef, useState } from "react";
import { useUpload } from "../hooks/useUpload";
import { ImageOut } from "../api/client";

export function ImageUploader({ onUploaded }: { onUploaded: (img: ImageOut) => void }) {
  const upload = useUpload();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const handle = useCallback(
    (file?: File) => {
      if (!file) return;
      upload.mutate(file, { onSuccess: onUploaded });
    },
    [upload, onUploaded],
  );

  return (
    <section aria-label="Upload">
      <div
        className="card"
        role="button"
        tabIndex={0}
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => e.key === "Enter" && inputRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          handle(e.dataTransfer.files?.[0]);
        }}
        style={{ textAlign: "center", outline: dragging ? "2px dashed var(--border)" : "none" }}
      >
        <p>Drop a character image here, or click to choose</p>
        <p className="muted">PNG / JPG</p>
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => handle(e.target.files?.[0])}
        />
      </div>
      {upload.isPending && <p className="muted">Uploading…</p>}
      {upload.isError && <p role="alert">{(upload.error as Error).message}</p>}
    </section>
  );
}
