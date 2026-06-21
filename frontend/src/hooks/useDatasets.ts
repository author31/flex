import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function useDatasets() {
  return useQuery({ queryKey: ["datasets"], queryFn: () => api.listDatasets() });
}

export function useDataset(datasetId: string | null) {
  return useQuery({
    queryKey: ["dataset", datasetId],
    queryFn: () => api.getDataset(datasetId!),
    enabled: !!datasetId,
  });
}
