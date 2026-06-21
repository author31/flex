import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, CreateDatasetBody } from "../api/client";

export function useUpdateDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ datasetId, body }: { datasetId: string; body: CreateDatasetBody }) =>
      api.updateDataset(datasetId, body),
    onSuccess: (ds) => {
      qc.invalidateQueries({ queryKey: ["datasets"] });
      qc.invalidateQueries({ queryKey: ["dataset", ds.dataset_id] });
    },
  });
}
