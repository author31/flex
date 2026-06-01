import { useQuery } from "@tanstack/react-query";
import { api, EditStatus } from "../api/client";

// Poll an edit job until it is completed or failed.
export function useEditPolling(editId: string | null) {
  return useQuery<EditStatus>({
    queryKey: ["edit", editId],
    queryFn: () => api.getEdit(editId as string),
    enabled: !!editId,
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === "completed" || s === "failed" ? false : 1500;
    },
  });
}
