import { useQuery } from "@tanstack/react-query";
import { api } from "../api/client";

export function usePresets() {
  return useQuery({ queryKey: ["presets"], queryFn: () => api.listPresets(), staleTime: Infinity });
}
