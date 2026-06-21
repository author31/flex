import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, CreateDatasetBody } from "../api/client";

export function useCreateDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateDatasetBody) => api.createDataset(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["datasets"] }),
  });
}
