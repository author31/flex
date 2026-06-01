import { useMutation } from "@tanstack/react-query";
import { api, CreateEditRequest } from "../api/client";

export function useCreateEdit() {
  return useMutation({ mutationFn: (body: CreateEditRequest) => api.createEdit(body) });
}
