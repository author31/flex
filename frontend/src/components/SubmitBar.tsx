export function SubmitBar({
  canSubmit,
  pending,
  onSubmit,
  error,
}: {
  canSubmit: boolean;
  pending: boolean;
  onSubmit: () => void;
  error?: string | null;
}) {
  return (
    <div style={{ marginTop: 16 }}>
      <button type="button" onClick={onSubmit} disabled={!canSubmit || pending}>
        {pending ? "Generating…" : "Generate edit"}
      </button>
      {!canSubmit && <span className="muted" style={{ marginLeft: 12 }}>Select a region and a prompt/preset</span>}
      {error && (
        <p role="alert" style={{ marginTop: 8 }}>
          {error}
        </p>
      )}
    </div>
  );
}
