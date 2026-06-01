import { useMutation } from "@tanstack/react-query";
import { api } from "../api/client";

export function useUpload() {
  return useMutation({ mutationFn: (file: File) => api.uploadImage(file) });
}
