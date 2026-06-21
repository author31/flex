import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => {
  const api = {
    listModels: vi.fn(async () => ({
      models: [
        { key: "base", model_id: "sdxl" },
        { key: "finetuned", model_id: "/data/ft" },
      ],
      default: "base",
    })),
    uploadImage: vi.fn(async () => ({
      image_id: "img_1",
      width: 128,
      height: 128,
      url: "/api/v1/images/img_1/file",
    })),
    createDataset: vi.fn(async () => ({
      dataset_id: "ds_1",
      name: "untitled study",
      model_key: "base",
      params: { steps: 25, guidance: 8 },
      items: [
        {
          id: "item_1",
          image_id: "img_1",
          image_url: "/api/v1/images/img_1/file",
          prompt: "a wide smile",
          position: 0,
          box: [20, 30, 60, 70],
        },
      ],
      revisions: [],
    })),
    listDatasets: vi.fn(async () => ({
      datasets: [{ dataset_id: "ds_saved", name: "prior", model_key: "base", item_count: 1, latest_revision: null }],
    })),
    getDataset: vi.fn(async () => ({
      dataset_id: "ds_saved",
      name: "prior",
      model_key: "finetuned",
      params: { steps: 30, guidance: 8 },
      items: [
        {
          id: "item_9",
          image_id: "img_9",
          image_url: "/api/v1/images/img_9/file",
          image_width: 100,
          image_height: 80,
          box: [4, 5, 20, 25],
          prompt: "loaded prompt",
          position: 0,
        },
      ],
      revisions: [],
    })),
    updateDataset: vi.fn(async () => ({
      dataset_id: "ds_saved",
      name: "prior",
      model_key: "finetuned",
      params: { steps: 30, guidance: 8 },
      items: [
        {
          id: "item_9",
          image_id: "img_9",
          image_url: "/api/v1/images/img_9/file",
          image_width: 100,
          image_height: 80,
          box: [4, 5, 20, 25],
          prompt: "loaded prompt edited",
          position: 0,
        },
      ],
      revisions: [],
    })),
    runBatch: vi.fn(async () => ({ revision_id: "rev_1", number: 1, status: "processing" })),
    getRevision: vi.fn(async () => ({
      dataset_id: "ds_1",
      number: 1,
      model_key: "base",
      params: { steps: 25, guidance: 8 },
      status: "completed",
      records: [
        {
          dataset_item_id: "item_1",
          status: "completed",
          result_image_id: "img_2",
          result_url: "/api/v1/images/img_2/file",
          metrics: {
            clip_similarity_in_mask: 0.31,
            edit_success: true,
            edit_success_score: 0.12,
            identity_preserved: true,
            latency_ms: 5,
          },
          error: null,
        },
      ],
    })),
  };
  return { api };
});

import { api } from "../api/client";
import { WorkspacePage } from "../pages/WorkspacePage";

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <WorkspacePage />
    </QueryClientProvider>,
  );
}

describe("workspace", () => {
  beforeEach(() => vi.clearAllMocks());

  it("upload → region + prompt → save → run → per-row metrics", async () => {
    const { container } = renderPage();
    const user = userEvent.setup();

    const file = new File(["x"], "c.png", { type: "image/png" });
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    // one row appears
    expect(await screen.findByTestId("workspace-row")).toBeInTheDocument();

    // drag a region on the row
    const surface = await screen.findByTestId("drag-surface");
    fireEvent.mouseDown(surface, { clientX: 20, clientY: 30 });
    fireEvent.mouseMove(surface, { clientX: 80, clientY: 100 });
    fireEvent.mouseUp(surface, { clientX: 80, clientY: 100 });

    // prompt
    await user.type(screen.getByLabelText("Prompt for image 1"), "a wide smile");

    // save dataset
    await user.click(screen.getByRole("button", { name: /save dataset/i }));
    expect(await screen.findByTestId("dataset-saved")).toHaveTextContent("ds_1");
    expect(api.createDataset).toHaveBeenCalledWith(
      expect.objectContaining({
        model_key: "base",
        items: [expect.objectContaining({ image_id: "img_1", box: [20, 30, 60, 70], prompt: "a wide smile" })],
      }),
    );

    // run batch → metrics fill in
    await user.click(screen.getByRole("button", { name: /run batch edit/i }));
    expect(await screen.findByTestId("metrics")).toBeInTheDocument();
    const row = screen.getByTestId("workspace-row");
    expect(within(row).getByText(/0.310/)).toBeInTheDocument();
    expect(within(row).getByText(/5 ms/)).toBeInTheDocument();
    // edited result image shown for the row
    const result = within(row).getByTestId("result-image") as HTMLImageElement;
    expect(result.src).toContain("/api/v1/images/img_2/file");
    expect(api.runBatch).toHaveBeenCalledWith("ds_1", "base");
  });

  it("loads a saved dataset and updates it in place", async () => {
    renderPage();
    const user = userEvent.setup();

    // wait for the library list, then load the saved dataset
    await screen.findByRole("option", { name: /prior/ });
    await user.selectOptions(screen.getByLabelText("Load saved dataset"), "ds_saved");

    // rows + prompt are populated from the saved dataset
    const prompt = (await screen.findByLabelText("Prompt for image 1")) as HTMLTextAreaElement;
    expect(prompt.value).toBe("loaded prompt");
    expect(api.getDataset).toHaveBeenCalledWith("ds_saved");

    // button switches to Update; edit the prompt and update in place
    const updateBtn = await screen.findByRole("button", { name: /update dataset/i });
    await user.clear(prompt);
    await user.type(prompt, "loaded prompt edited");
    await user.click(updateBtn);

    expect(api.updateDataset).toHaveBeenCalledWith(
      "ds_saved",
      expect.objectContaining({
        model_key: "finetuned",
        items: [expect.objectContaining({ image_id: "img_9", prompt: "loaded prompt edited" })],
      }),
    );
    expect(api.createDataset).not.toHaveBeenCalled();
  });

  it("blocks save until every row has a region and prompt", async () => {
    const { container } = renderPage();
    const user = userEvent.setup();
    const file = new File(["x"], "c.png", { type: "image/png" });
    await user.upload(container.querySelector('input[type="file"]') as HTMLInputElement, file);
    expect(await screen.findByRole("button", { name: /save dataset/i })).toBeDisabled();
  });
});
