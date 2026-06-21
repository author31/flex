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
  model?: string; // checkpoint key (see listModels); omit for default
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

export interface MeshStatus {
  mesh_id: string;
  status: JobStatus;
  url?: string | null;
  error?: string | null;
}

// --- Study workspace (feature 002) ----------------------------------------- //

export interface ModelInfo {
  key: string;
  model_id: string;
  lora?: string | null; // adapter dir if this checkpoint has a LoRA
}

export interface ModelsOut {
  models: ModelInfo[];
  default: string;
}

export interface DatasetItemIn {
  image_id: string;
  region?: FacialRegion;
  box?: [number, number, number, number];
  prompt: string;
}

export interface CreateDatasetBody {
  name?: string;
  model_key: string;
  params?: EditParams;
  items: DatasetItemIn[];
}

export interface DatasetItemOut {
  id: string;
  image_id: string;
  image_url: string;
  image_width?: number | null;
  image_height?: number | null;
  region?: FacialRegion | null;
  box?: [number, number, number, number] | null;
  prompt: string;
  position: number;
}

export interface RevisionSummary {
  number: number;
  model_key: string;
  status: JobStatus;
  created_at?: string | null;
}

export interface DatasetSummary {
  dataset_id: string;
  name: string;
  model_key: string;
  item_count: number;
  latest_revision: number | null;
}

export interface DatasetOut {
  dataset_id: string;
  name: string;
  model_key: string;
  params: EditParams;
  items: DatasetItemOut[];
  revisions: RevisionSummary[];
}

export interface MetricRecordOut {
  dataset_item_id: string;
  status: JobStatus;
  result_image_id?: string | null;
  result_url?: string | null;
  metrics?: Metrics | null;
  error?: string | null;
}

export interface RevisionOut {
  dataset_id: string;
  number: number;
  model_key: string;
  params: EditParams;
  status: JobStatus;
  created_at?: string | null;
  records: MetricRecordOut[];
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

  // Expert: 3D mesh export
  async createMesh(editId: string): Promise<{ mesh_id: string; status: JobStatus }> {
    return json(await fetch(`${BASE}/edits/${editId}/mesh`, { method: "POST" }));
  },

  async getMesh(meshId: string): Promise<MeshStatus> {
    return json(await fetch(`${BASE}/mesh/${meshId}`));
  },

  // --- Study workspace (feature 002) --------------------------------------- //

  async listModels(): Promise<ModelsOut> {
    return json(await fetch(`${BASE}/models`));
  },

  async createDataset(body: CreateDatasetBody): Promise<DatasetOut> {
    return json(
      await fetch(`${BASE}/datasets`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },

  async listDatasets(): Promise<{ datasets: DatasetSummary[] }> {
    return json(await fetch(`${BASE}/datasets`));
  },

  async getDataset(datasetId: string): Promise<DatasetOut> {
    return json(await fetch(`${BASE}/datasets/${datasetId}`));
  },

  async updateDataset(datasetId: string, body: CreateDatasetBody): Promise<DatasetOut> {
    return json(
      await fetch(`${BASE}/datasets/${datasetId}`, {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(body),
      }),
    );
  },

  async runBatch(
    datasetId: string,
    modelKey?: string,
  ): Promise<{ revision_id: string; number: number; status: JobStatus }> {
    return json(
      await fetch(`${BASE}/datasets/${datasetId}/runs`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(modelKey ? { model_key: modelKey } : {}),
      }),
    );
  },

  async getRevision(datasetId: string, number: number): Promise<RevisionOut> {
    return json(await fetch(`${BASE}/datasets/${datasetId}/revisions/${number}`));
  },
};
