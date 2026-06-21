import { useMutation } from "@tanstack/react-query";
import { api } from "../api/client";

export function useRunBatch() {
  return useMutation({
    mutationFn: ({ datasetId, modelKey }: { datasetId: string; modelKey?: string }) =>
      api.runBatch(datasetId, modelKey),
  });
}
