import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../api/client", () => {
  const api = {
    base: "/api/v1",
    fileUrl: (p: string) => p,
    listPresets: vi.fn(async () => ({
      expressions: ["smile"],
      regions: ["face", "custom"],
    })),
    listModels: vi.fn(async () => ({ models: [{ key: "base", model_id: "sdxl" }], default: "base" })),
    listDatasets: vi.fn(async () => ({ datasets: [] })),
  };
  return { api };
});

import { App } from "../App";

function renderApp() {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={qc}>
      <App />
    </QueryClientProvider>,
  );
}

describe("routing", () => {
  beforeEach(() => vi.clearAllMocks());

  it("nav switches between editor, workspace and compare", async () => {
    renderApp();
    const user = userEvent.setup();

    // default route → editor
    expect(screen.getByRole("heading", { name: /facial expression editor/i })).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: "Workspace" }));
    expect(await screen.findByRole("heading", { name: /study workspace/i })).toBeInTheDocument();

    await user.click(screen.getByRole("link", { name: "Compare" }));
    expect(await screen.findByRole("heading", { name: /compare models/i })).toBeInTheDocument();
  });
});
