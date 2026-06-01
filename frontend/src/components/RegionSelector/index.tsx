import { ImageOut } from "../../api/client";
import { DragSelect } from "./DragSelect";

export type Selection = { box: [number, number, number, number] } | null;

// Region selection: drag a rectangle over the uploaded character.
export function RegionSelector({
  image,
  onSelection,
}: {
  image: ImageOut;
  onSelection: (s: Selection) => void;
}) {
  return (
    <section className="card" aria-label="Select region">
      <DragSelect
        imageUrl={image.url}
        width={image.width}
        height={image.height}
        onChange={(box) => onSelection(box ? { box } : null)}
      />
    </section>
  );
}
