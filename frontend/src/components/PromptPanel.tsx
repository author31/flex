import { useState } from "react";
import { EditParams, ExpressionPreset } from "../api/client";
import { usePresets } from "../hooks/usePresets";

export interface PromptState {
  prompt?: string;
  preset?: ExpressionPreset;
  params: EditParams;
}

export function PromptPanel({ onChange }: { onChange: (s: PromptState) => void }) {
  const presets = usePresets();
  const [preset, setPreset] = useState<ExpressionPreset | null>(null);
  const [prompt, setPrompt] = useState("");
  const [params, setParams] = useState<EditParams>({ strength: 0.99, steps: 25, guidance: 8.0 });
  const [advanced, setAdvanced] = useState(false);

  function emit(next: Partial<PromptState>) {
    const state: PromptState = {
      preset: next.preset !== undefined ? next.preset ?? undefined : preset ?? undefined,
      prompt: next.prompt !== undefined ? next.prompt : prompt,
      params: next.params ?? params,
    };
    // exactly one of preset|prompt — preset wins when set
    if (state.preset) state.prompt = undefined;
    onChange(state);
  }

  return (
    <section className="card" aria-label="Prompt">
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        {(presets.data?.expressions ?? ["smile", "angry", "sad", "surprised"]).map((p) => (
          <button
            key={p}
            type="button"
            aria-pressed={preset === p}
            onClick={() => {
              const next = preset === p ? null : (p as ExpressionPreset);
              setPreset(next);
              setPrompt("");
              emit({ preset: next ?? undefined, prompt: "" });
            }}
          >
            {p}
          </button>
        ))}
      </div>

      <p className="secondary" style={{ margin: "12px 0 4px" }}>
        …or describe the edit
      </p>
      <input
        type="text"
        aria-label="Prompt text"
        placeholder="e.g. a wide happy smile"
        value={prompt}
        onChange={(e) => {
          setPrompt(e.target.value);
          setPreset(null);
          emit({ prompt: e.target.value, preset: undefined });
        }}
        style={{ width: "100%" }}
      />

      <button type="button" className="muted" onClick={() => setAdvanced((v) => !v)} style={{ marginTop: 12 }}>
        {advanced ? "Hide" : "Advanced"} params
      </button>
      {advanced && (
        <div style={{ display: "flex", gap: 12, marginTop: 8, flexWrap: "wrap" }}>
          {(["strength", "steps", "guidance", "seed"] as const).map((k) => (
            <label key={k} className="muted">
              {k}
              <input
                type="number"
                step={k === "strength" ? 0.01 : 1}
                value={(params as Record<string, number | undefined>)[k] ?? ""}
                onChange={(e) => {
                  const v = e.target.value === "" ? undefined : Number(e.target.value);
                  const np = { ...params, [k]: v };
                  setParams(np);
                  emit({ params: np });
                }}
                style={{ width: 90, display: "block" }}
              />
            </label>
          ))}
        </div>
      )}
    </section>
  );
}
