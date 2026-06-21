import { useMutation, useQuery } from "@tanstack/react-query";
import { api, MeshStatus } from "../api/client";

export function useCreateMesh() {
  return useMutation({ mutationFn: (editId: string) => api.createMesh(editId) });
}

// Poll a mesh job until completed or failed.
export function useMeshPolling(meshId: string | null) {
  return useQuery<MeshStatus>({
    queryKey: ["mesh", meshId],
    queryFn: () => api.getMesh(meshId as string),
    enabled: !!meshId,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "completed" || s === "failed" ? false : 1500;
    },
  });
}
