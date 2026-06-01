import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DragSelect } from "../RegionSelector/DragSelect";

describe("DragSelect", () => {
  it("emits a box in image-pixel space after a drag", () => {
    const onChange = vi.fn();
    render(<DragSelect imageUrl="/u" width={128} height={128} onChange={onChange} />);
    const surface = screen.getByTestId("drag-surface");

    fireEvent.mouseDown(surface, { clientX: 10, clientY: 10 });
    fireEvent.mouseMove(surface, { clientX: 60, clientY: 80 });
    fireEvent.mouseUp(surface, { clientX: 60, clientY: 80 });

    // jsdom getBoundingClientRect is 0-sized → scale 1:1, so box == local coords.
    expect(onChange).toHaveBeenLastCalledWith([10, 10, 50, 70]);
  });

  it("emits null for a too-small drag (a click)", () => {
    const onChange = vi.fn();
    render(<DragSelect imageUrl="/u" width={128} height={128} onChange={onChange} />);
    const surface = screen.getByTestId("drag-surface");

    fireEvent.mouseDown(surface, { clientX: 10, clientY: 10 });
    fireEvent.mouseUp(surface, { clientX: 11, clientY: 11 });

    expect(onChange).toHaveBeenLastCalledWith(null);
  });
});
