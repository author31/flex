import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

// Poll a revision until it leaves "processing" so per-row metrics fill in live.
export function useRevision(datasetId: string | null, number: number | null) {
  return useQuery({
    queryKey: ["revision", datasetId, number],
    queryFn: () => api.getRevision(datasetId!, number!),
    enabled: !!datasetId && number != null,
    refetchInterval: (q) =>
      q.state.data && q.state.data.status === "processing" ? 1500 : false,
  });
}
