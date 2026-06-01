import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ComparisonView } from "../ComparisonView";
import { Comparison } from "../../api/client";

const comparison: Comparison = {
  edit_id: "edit_1",
  status: "completed",
  original: { image_id: "img_1", url: "/api/v1/images/img_1/file" },
  edited: { image_id: "img_2", url: "/api/v1/images/img_2/file" },
  region: "mouth",
  prompt: "a wide happy smile",
  metrics: {
    clip_similarity_in_mask: 0.314,
    edit_success: true,
    edit_success_score: 0.12,
    identity_preserved: true,
    latency_ms: 4200,
  },
};

describe("ComparisonView", () => {
  it("renders original + edited images and metrics", () => {
    render(<ComparisonView comparison={comparison} />);
    expect(screen.getByAltText("original")).toHaveAttribute("src", comparison.original.url);
    expect(screen.getByAltText("edited")).toHaveAttribute("src", comparison.edited.url);
    expect(screen.getByText("0.314")).toBeInTheDocument();
    expect(screen.getByText("4200 ms")).toBeInTheDocument();
    expect(screen.getByLabelText("Metrics")).toBeInTheDocument();
  });

  it("shows the prompt and region", () => {
    render(<ComparisonView comparison={comparison} />);
    expect(screen.getByText(/a wide happy smile/)).toBeInTheDocument();
    expect(screen.getByText(/mouth/)).toBeInTheDocument();
  });
});
