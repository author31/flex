import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MeshViewer } from "../MeshViewer";

describe("MeshViewer", () => {
  it("renders a model-viewer pointing at the mesh url", () => {
    render(<MeshViewer url="/api/v1/mesh/mesh_1/file" />);
    const viewer = screen.getByTestId("mesh-viewer");
    expect(viewer).toHaveAttribute("src", "/api/v1/mesh/mesh_1/file");
    expect(screen.getByText(/download .glb/i)).toHaveAttribute("href", "/api/v1/mesh/mesh_1/file");
  });
});
