import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

const dataset = {
  dataset_id: "ds_1",
  name: "smiles",
  model_key: "base",
  params: { steps: 25, guidance: 8 },
  items: [
    { id: "item_1", image_id: "img_1", image_url: "/api/v1/images/img_1/file", prompt: "smile", position: 0 },
    { id: "item_2", image_id: "img_2", image_url: "/api/v1/images/img_2/file", prompt: "grin", position: 1 },
  ],
  revisions: [],
};

const metric = (id: string, score: number) => ({
  dataset_item_id: id,
  status: "completed",
  result_image_id: "out",
  result_url: `/api/v1/images/out_${id}/file`,
  metrics: {
    clip_similarity_in_mask: score,
    edit_success: true,
    edit_success_score: 0.1,
    identity_preserved: true,
    latency_ms: 5,
  },
  error: null,
});

vi.mock("../api/client", () => {
  const api = {
    listModels: vi.fn(async () => ({
      models: [
        { key: "base", model_id: "sdxl" },
        { key: "finetuned", model_id: "/data/ft" },
      ],
      default: "base",
    })),
    listDatasets: vi.fn(async () => ({
      datasets: [{ dataset_id: "ds_1", name: "smiles", model_key: "base", item_count: 2, latest_revision: null }],
    })),
    getDataset: vi.fn(async () => dataset),
    runBatch: vi.fn(async (_id: string, key?: string) => ({
      revision_id: `rev_${key}`,
      number: key === "base" ? 1 : 2,
      status: "processing",
    })),
    getRevision: vi.fn(async (_id: string, n: number) => ({
      dataset_id: "ds_1",
      number: n,
      model_key: n === 1 ? "base" : "finetuned",
      params: { steps: 25, guidance: 8 },
      status: "completed",
      records: [metric("item_1", n === 1 ? 0.31 : 0.42), metric("item_2", n === 1 ? 0.2 : 0.5)],
    })),
  };
  return { api };
});

import { api } from "../api/client";
import { ComparePage } from "../pages/ComparePage";

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <ComparePage />
    </QueryClientProvider>,
  );
}

describe("compare", () => {
  beforeEach(() => vi.clearAllMocks());

  it("select dataset → run both models → row-aligned results per column", async () => {
    renderPage();
    const user = userEvent.setup();

    await screen.findByRole("option", { name: /smiles/ }); // wait for dataset list
    await user.selectOptions(screen.getByLabelText("Dataset"), "ds_1");

    const colA = await screen.findByLabelText("Model A");
    const colB = await screen.findByLabelText("Model B");
    expect(colA).toBeInTheDocument();
    expect(colB).toBeInTheDocument();

    // default selections: A=base, B=finetuned
    await user.click(within(colA.closest("section")!).getByRole("button", { name: "Run" }));
    await user.click(within(colB.closest("section")!).getByRole("button", { name: "Run" }));

    // both columns show two aligned result rows
    expect(await screen.findAllByTestId("compare-row")).toHaveLength(4);
    expect(api.runBatch).toHaveBeenCalledWith("ds_1", "base");
    expect(api.runBatch).toHaveBeenCalledWith("ds_1", "finetuned");
    // finetuned (B) result metric visible
    expect(await screen.findByText(/0.420/)).toBeInTheDocument();
  });
});
