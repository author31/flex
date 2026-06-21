import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { RegionPreview } from "../RegionPreview";

const image = {
  image_id: "img_1",
  width: 128,
  height: 128,
  url: "/api/v1/images/img_1/file",
};

describe("RegionPreview", () => {
  it("shows the selected region crop + dimensions", () => {
    render(<RegionPreview image={image} box={[20, 30, 40, 50]} />);
    const crop = screen.getByTestId("region-preview");
    expect(crop.style.backgroundImage).toContain("img_1/file");
    // scale = min(1, 200/40) = 1 → crop offset == box origin
    expect(crop.style.backgroundPosition).toBe("-20px -30px");
    expect(screen.getByText(/40 × 50 px/)).toBeInTheDocument();
  });
});
