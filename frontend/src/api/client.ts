// Typed API client. Mirrors contracts/openapi.yaml. Base = VITE_API_BASE (default /api).

const BASE = `${import.meta.env.VITE_API_BASE ?? "/api"}/v1`;

export type FacialRegion = "face" | "eyes" | "mouth" | "eyebrows" | "custom";
export type ExpressionPreset = "smile" | "angry" | "sad" | "surprised";
export type JobStatus = "processing" | "completed" | "failed";

export interface ImageOut {
  image_id: string;
  width: number;
  height: number;
  url: string;
}

export interface RegionMask {
  url: string;
  bbox: [number, number, number, number];
  area: number;
}

export interface SegmentsOut {
  regions: Partial<Record<FacialRegion, RegionMask>>;
}

export interface EditParams {
  strength?: number;
  steps?: number;
  guidance?: number;
  seed?: number | null;
}

export interface CreateEditRequest {
  image_id: string;
  box?: [number, number, number, number]; // dragged rectangle [x, y, w, h] (primary UX)
  region?: FacialRegion;
  mask?: string;
  prompt?: string;
  preset?: ExpressionPreset;
  params?: EditParams;
}

export interface Metrics {
  clip_similarity_in_mask: number;
  edit_success: boolean;
  edit_success_score: number;
  identity_preserved: boolean;
  latency_ms: number;
}

export interface EditStatus {
  edit_id: string;
  status: JobStatus;
  result?: { result_image_id: string; url: string } | null;
  metrics?: Metrics | null;
  error?: string | null;
}

export interface Comparison {
  edit_id: string;
  status: JobStatus;
  original: { image_id: string; url: string };
  edited: { image_id: string; url: string };
  region: FacialRegion;
  prompt: string;
  metrics: Metrics;
}

export interface Presets {
  expressions: ExpressionPreset[];
  regions: FacialRegion[];
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  base: BASE,
  fileUrl: (path: string) => path, // backend returns absolute /api/v1/... paths

  async uploadImage(file: File): Promise<ImageOut> {
    const fd = new FormData();
    fd.append("file", file);
    return json(await fetch(`${BASE}/images`, { method: "POST", body: fd }));
  },

  async segment(imageId: string): Promise<SegmentsOut> {
    return json(await fetch(`${BASE}/images/${imageId}/segments`, { method: "POST" }));
  },

  async listPresets(): Promise<Presets> {
    return json(await fetch(`${BASE}/presets`));
  },

  async createEdit(body: CreateEditRequest): Promise<{ edit_id: string; status: JobStatus }> {
    return json(
      await fetch(`${BASE}/edits`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },

  async getEdit(editId: string): Promise<EditStatus> {
    return json(await fetch(`${BASE}/edits/${editId}`));
  },

  async getComparison(editId: string): Promise<Comparison> {
    return json(await fetch(`${BASE}/edits/${editId}/comparison`));
  },
};
