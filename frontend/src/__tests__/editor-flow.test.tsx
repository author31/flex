import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

// Mock the API client module with an in-test fake.
vi.mock("../api/client", () => {
  const api = {
    base: "/api/v1",
    fileUrl: (p: string) => p,
    uploadImage: vi.fn(async () => ({
      image_id: "img_1",
      width: 128,
      height: 128,
      url: "/api/v1/images/img_1/file",
    })),
    listPresets: vi.fn(async () => ({
      expressions: ["smile", "angry", "sad", "surprised"],
      regions: ["face", "eyes", "mouth", "eyebrows", "custom"],
    })),
    createEdit: vi.fn(async () => ({ edit_id: "edit_1", status: "processing" })),
    getEdit: vi.fn(async () => ({
      edit_id: "edit_1",
      status: "completed",
      result: { result_image_id: "img_2", url: "/api/v1/images/img_2/file" },
    })),
    getComparison: vi.fn(async () => ({
      edit_id: "edit_1",
      status: "completed",
      original: { image_id: "img_1", url: "/api/v1/images/img_1/file" },
      edited: { image_id: "img_2", url: "/api/v1/images/img_2/file" },
      region: "custom",
      prompt: "smile",
      metrics: {
        clip_similarity_in_mask: 0.31,
        edit_success: true,
        edit_success_score: 0.12,
        identity_preserved: true,
        latency_ms: 5,
      },
    })),
  };
  return { api };
});

import { api } from "../api/client";
import { EditorPage } from "../pages/EditorPage";

function renderPage() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <EditorPage />
    </QueryClientProvider>,
  );
}

describe("editor flow", () => {
  beforeEach(() => vi.clearAllMocks());

  it("upload → drag region → preset → submit → comparison + metrics", async () => {
    const { container } = renderPage();
    const user = userEvent.setup();

    // upload
    const file = new File(["x"], "c.png", { type: "image/png" });
    const input = container.querySelector('input[type="file"]') as HTMLInputElement;
    await user.upload(input, file);

    // drag a rectangle over the image
    const surface = await screen.findByTestId("drag-surface");
    fireEvent.mouseDown(surface, { clientX: 20, clientY: 30 });
    fireEvent.mouseMove(surface, { clientX: 80, clientY: 100 });
    fireEvent.mouseUp(surface, { clientX: 80, clientY: 100 });

    // preset
    await user.click(await screen.findByRole("button", { name: "smile" }));

    // submit
    await user.click(screen.getByRole("button", { name: /generate edit/i }));

    // comparison appears with metrics
    expect(await screen.findByLabelText("Comparison", {}, { timeout: 4000 })).toBeInTheDocument();
    expect(await screen.findByLabelText("Metrics")).toBeInTheDocument();
    expect(screen.getByText("0.310")).toBeInTheDocument();

    // the request carried the dragged box
    expect(api.createEdit).toHaveBeenCalledWith(
      expect.objectContaining({ image_id: "img_1", box: [20, 30, 60, 70], preset: "smile" }),
    );
  });
});
